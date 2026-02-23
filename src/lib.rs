use parking_lot::{Condvar, Mutex};
use pyo3::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use windows::Win32::Foundation::HMODULE;
use windows::Win32::Foundation::{LPARAM, LUID, WPARAM};
use windows::Win32::Graphics::Direct3D::{D3D_DRIVER_TYPE_UNKNOWN, D3D_FEATURE_LEVEL_11_1};
use windows::Win32::Graphics::Direct3D11::{
    D3D11_BIND_SHADER_RESOURCE, D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_RESOURCE_MISC_SHARED,
    D3D11_SDK_VERSION, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT, D3D11CreateDevice, ID3D11Device,
    ID3D11DeviceContext, ID3D11Texture2D,
};
use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1};
use windows::Win32::System::Threading::GetCurrentThreadId;
use windows::Win32::System::WinRT::{
    CreateDispatcherQueueController, DQTAT_COM_NONE, DQTYPE_THREAD_CURRENT, DispatcherQueueOptions,
};
use windows::Win32::UI::WindowsAndMessaging::{
    DispatchMessageW, GetMessageW, MSG, PostThreadMessageW, TranslateMessage, WM_QUIT,
};
use windows::core::Interface;

use windows_capture::capture::{Context, GraphicsCaptureApiHandler};
use windows_capture::frame::Frame;
use windows_capture::graphics_capture_api::{GraphicsCaptureApi, InternalCaptureControl};
use windows_capture::monitor::Monitor;
use windows_capture::settings::{
    ColorFormat, CursorCaptureSettings, DirtyRegionSettings, DrawBorderSettings,
    GraphicsCaptureItemType, MinimumUpdateIntervalSettings, SecondaryWindowSettings,
};
use windows_capture::window::Window;

#[derive(thiserror::Error, Debug)]
pub enum WcCudaError {
    #[error("Windows API error: {0}")]
    Windows(#[from] windows::core::Error),
    #[error("AdapterNotFound")]
    AdapterNotFound,
    #[error("Python error: {0}")]
    Python(#[from] PyErr),
    #[error("Capture error: {0}")]
    Capture(String),
    #[error("Item conversion failed")]
    ItemConvertFailed,
}

impl From<WcCudaError> for PyErr {
    fn from(err: WcCudaError) -> PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
    }
}

pub fn create_d3d_device_with_luid(
    luid: Option<LUID>,
) -> Result<(ID3D11Device, ID3D11DeviceContext), WcCudaError> {
    let adapter = if let Some(luid) = luid {
        let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1()? };
        let mut adapter = None;
        let mut i = 0;
        loop {
            let current_adapter = match unsafe { factory.EnumAdapters1(i) } {
                Ok(a) => a,
                Err(_) => break,
            };
            let desc = unsafe { current_adapter.GetDesc1()? };
            if desc.AdapterLuid.LowPart == luid.LowPart
                && desc.AdapterLuid.HighPart == luid.HighPart
            {
                adapter = Some(current_adapter);
                break;
            }
            i += 1;
        }
        adapter.ok_or(WcCudaError::AdapterNotFound)?
    } else {
        let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1()? };
        unsafe { factory.EnumAdapters1(0)? }
    };

    let mut d3d_device = None;
    let mut d3d_device_context = None;
    let feature_levels = [D3D_FEATURE_LEVEL_11_1];

    unsafe {
        D3D11CreateDevice(
            &adapter,
            D3D_DRIVER_TYPE_UNKNOWN,
            HMODULE::default(),
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            Some(&feature_levels),
            D3D11_SDK_VERSION,
            Some(&mut d3d_device),
            None,
            Some(&mut d3d_device_context),
        )?;
    }

    Ok((d3d_device.unwrap(), d3d_device_context.unwrap()))
}

#[pyclass]
#[derive(Clone)]
pub struct WcGpuFrame {
    pub texture: ID3D11Texture2D,
    #[pyo3(get)]
    pub width: u32,
    #[pyo3(get)]
    pub height: u32,
    #[pyo3(get)]
    pub original_width: u32,
    #[pyo3(get)]
    pub original_height: u32,
}

#[pymethods]
impl WcGpuFrame {
    #[getter]
    fn texture_ptr(&self) -> usize {
        self.texture.as_raw() as usize
    }
}

struct FrameSlot {
    data: Mutex<(Option<WcGpuFrame>, u64)>,
    condvar: Condvar,
}

struct InnerHandler {
    shared_slot: Arc<FrameSlot>,
    shared_texture: Option<ID3D11Texture2D>,
    current_aligned_width: u32,
    current_aligned_height: u32,
    thread_id: u32,
}

impl GraphicsCaptureApiHandler for InnerHandler {
    type Flags = (Arc<FrameSlot>, u32);
    type Error = WcCudaError;

    fn new(ctx: Context<Self::Flags>) -> Result<Self, Self::Error> {
        Ok(Self {
            shared_slot: ctx.flags.0,
            thread_id: ctx.flags.1,
            shared_texture: None,
            current_aligned_width: 0,
            current_aligned_height: 0,
        })
    }

    fn on_frame_arrived(
        &mut self,
        frame: &mut Frame,
        _capture_control: InternalCaptureControl,
    ) -> Result<(), Self::Error> {
        let device = frame.device();
        let context = frame.device_context();
        let width = frame.width();
        let height = frame.height();
        let format = frame.desc().Format;

        let aligned_width = (width + 63) & !63;
        let aligned_height = (height + 31) & !31;

        let recreate = self.shared_texture.is_none()
            || self.current_aligned_width != aligned_width
            || self.current_aligned_height != aligned_height;

        if recreate {
            let desc = D3D11_TEXTURE2D_DESC {
                Width: aligned_width,
                Height: aligned_height,
                MipLevels: 1,
                ArraySize: 1,
                Format: format,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
                CPUAccessFlags: 0,
                MiscFlags: D3D11_RESOURCE_MISC_SHARED.0 as u32,
            };
            let mut tex = None;
            unsafe { device.CreateTexture2D(&desc, None, Some(&mut tex))? };
            self.shared_texture = Some(tex.unwrap());
            self.current_aligned_width = aligned_width;
            self.current_aligned_height = aligned_height;
        }

        if let Some(ref tex) = self.shared_texture {
            unsafe {
                context.CopySubresourceRegion(tex, 0, 0, 0, 0, frame.as_raw_texture(), 0, None);
                context.Flush();
            }
            let gpu_frame = WcGpuFrame {
                texture: tex.clone(),
                width: aligned_width,
                height: aligned_height,
                original_width: width,
                original_height: height,
            };
            
            // Update the shared slot with the latest frame
            let mut data = self.shared_slot.data.lock();
            data.0 = Some(gpu_frame);
            data.1 += 1;
            self.shared_slot.condvar.notify_all();
        }
        Ok(())
    }

    fn on_closed(&mut self) -> Result<(), Self::Error> {
        unsafe {
            let _ = PostThreadMessageW(self.thread_id, WM_QUIT, WPARAM(0), LPARAM(0));
        }
        Ok(())
    }
}

#[pyclass]
pub struct WcCapture {
    luid: Option<(u32, i32)>,
    monitor_index: Option<usize>,
    window_hwnd: Option<isize>,
    window_title: Option<String>,
    shared_slot: Arc<FrameSlot>,
    thread_handle: Mutex<Option<thread::JoinHandle<()>>>,
    thread_id: Arc<Mutex<Option<u32>>>,
    should_stop: Arc<AtomicBool>,
    last_error: Arc<Mutex<Option<String>>>,
}

#[pymethods]
impl WcCapture {
    #[new]
    #[pyo3(signature = (luid=None, monitor_index=None, window_hwnd=None, window_title=None))]
    fn new(
        luid: Option<(u32, i32)>,
        monitor_index: Option<usize>,
        window_hwnd: Option<isize>,
        window_title: Option<String>,
    ) -> Self {
        Self {
            luid,
            monitor_index,
            window_hwnd,
            window_title,
            shared_slot: Arc::new(FrameSlot {
                data: Mutex::new((None, 0)),
                condvar: Condvar::new(),
            }),
            thread_handle: Mutex::new(None),
            thread_id: Arc::new(Mutex::new(Option::None)),
            should_stop: Arc::new(AtomicBool::new(false)),
            last_error: Arc::new(Mutex::new(None)),
        }
    }

    fn start(&self) -> PyResult<()> {
        let mut handle_lock = self.thread_handle.lock();
        if handle_lock.is_some() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Capture already running",
            ));
        }

        let luid = self.luid.map(|(low, high)| LUID {
            LowPart: low,
            HighPart: high,
        });
        let monitor_index = self.monitor_index;
        let window_hwnd = self.window_hwnd;
        let window_title = self.window_title.clone();
        let thread_id_ptr = self.thread_id.clone();
        let should_stop = self.should_stop.clone();
        let last_error = self.last_error.clone();
        let shared_slot = self.shared_slot.clone();

        should_stop.store(false, Ordering::SeqCst);
        *last_error.lock() = None;

        let handle = thread::spawn(move || {
            let res: Result<(), WcCudaError> = (|| {
                if should_stop.load(Ordering::SeqCst) {
                    return Ok(());
                }

                let (d3d_device, d3d_device_context) = create_d3d_device_with_luid(luid)?;

                let item_type: GraphicsCaptureItemType = if let Some(title) = window_title {
                    Window::from_contains_name(&title)
                        .map_err(|_| WcCudaError::Capture(format!("Window not found: {}", title)))?
                        .try_into()
                        .map_err(|_| WcCudaError::ItemConvertFailed)?
                } else if let Some(hwnd) = window_hwnd {
                    Window::from_raw_hwnd(hwnd as *mut std::ffi::c_void)
                        .try_into()
                        .map_err(|_| WcCudaError::ItemConvertFailed)?
                } else {
                    let idx = monitor_index.unwrap_or(1);
                    Monitor::from_index(idx)
                        .map_err(|_| WcCudaError::ItemConvertFailed)?
                        .try_into()
                        .map_err(|_| WcCudaError::ItemConvertFailed)?
                };

                let options = DispatcherQueueOptions {
                    dwSize: std::mem::size_of::<DispatcherQueueOptions>() as u32,
                    threadType: DQTYPE_THREAD_CURRENT,
                    apartmentType: DQTAT_COM_NONE,
                };
                let _controller = unsafe { CreateDispatcherQueueController(options)? };
                let tid = unsafe { GetCurrentThreadId() };
                *thread_id_ptr.lock() = Some(tid);

                if should_stop.load(Ordering::SeqCst) {
                    return Ok(());
                }

                let ctx = Context {
                    flags: (shared_slot, tid),
                    device: d3d_device.clone(),
                    device_context: d3d_device_context.clone(),
                };
                let handler = InnerHandler::new(ctx)?;
                let callback_arc = Arc::new(Mutex::new(handler));

                let result = Arc::new(Mutex::new(None));
                let mut capture = GraphicsCaptureApi::new(
                    d3d_device,
                    d3d_device_context,
                    item_type,
                    callback_arc,
                    CursorCaptureSettings::Default,
                    DrawBorderSettings::Default,
                    SecondaryWindowSettings::Default,
                    MinimumUpdateIntervalSettings::Default,
                    DirtyRegionSettings::Default,
                    ColorFormat::Bgra8,
                    tid,
                    result,
                )
                .map_err(|e| WcCudaError::Capture(e.to_string()))?;

                capture
                    .start_capture()
                    .map_err(|e| WcCudaError::Capture(e.to_string()))?;

                let mut message = MSG::default();
                unsafe {
                    while GetMessageW(&mut message, None, 0, 0).as_bool() {
                        let _ = TranslateMessage(&message);
                        DispatchMessageW(&message);
                    }
                }
                Ok(())
            })();
            
            if let Err(e) = res {
                *last_error.lock() = Some(e.to_string());
            }
            *thread_id_ptr.lock() = None;
        });
        *handle_lock = Some(handle);
        Ok(())
    }

    fn stop(&self) {
        self.should_stop.store(true, Ordering::SeqCst);
        if let Some(tid) = *self.thread_id.lock() {
            unsafe {
                let _ = PostThreadMessageW(tid, WM_QUIT, WPARAM(0), LPARAM(0));
            }
        }

        let handle = self.thread_handle.lock().take();
        if let Some(h) = handle {
            if h.thread().id() != thread::current().id() {
                let _ = h.join();
            }
        }
        let mut data = self.shared_slot.data.lock();
        data.0 = None;
    }

    fn is_alive(&self) -> bool {
        let tid = self.thread_id.lock();
        let handle = self.thread_handle.lock();
        handle.is_some() && tid.is_some()
    }

    fn get_last_error(&self) -> Option<String> {
        self.last_error.lock().clone()
    }

    #[pyo3(signature = (last_id, timeout=None))]
    fn get_frame(&self, py: Python, last_id: u64, timeout: Option<f32>) -> Option<(WcGpuFrame, u64)> {
        let slot = self.shared_slot.clone();
        py.allow_threads(move || {
            let mut data = slot.data.lock();
            if data.1 <= last_id && timeout.is_some() {
                let dur = Duration::from_secs_f32(timeout.unwrap());
                let _ = slot.condvar.wait_for(&mut data, dur);
            }
            
            if data.1 > last_id {
                data.0.as_ref().map(|f| (f.clone(), data.1))
            } else {
                None
            }
        })
    }
}

#[pymodule]
fn _wc_cuda(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<WcCapture>()?;
    m.add_class::<WcGpuFrame>()?;
    Ok(())
}
