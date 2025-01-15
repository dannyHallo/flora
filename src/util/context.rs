#![allow(unused)]

use std::sync::Arc;

use vulkano::{
    device::{physical::PhysicalDeviceType, Device, Queue},
    instance::{Instance, InstanceExtensions},
    memory::allocator::StandardMemoryAllocator,
    swapchain::Surface,
    sync::event,
};

use winit::event_loop::EventLoop;

use vulkano::VulkanLibrary;

fn create_instance(
    config: &VulkanoConfig,
    lib: Arc<VulkanLibrary>,
    event_loop: Option<&EventLoop<()>>,
) -> Arc<Instance> {
    use vulkano::instance::{InstanceCreateFlags, InstanceCreateInfo};

    let surface_extensions = {
        match event_loop {
            Some(event_loop) => Surface::required_extensions(event_loop),
            None => InstanceExtensions::empty(),
        }
    };
    // https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_portability_enumeration.html
    let portability_extensions = InstanceExtensions {
        khr_portability_enumeration: true,
        ..InstanceExtensions::empty()
    };
    let all_extensions = surface_extensions.union(&portability_extensions);

    // print
    println!("surface extensions: {:?}", all_extensions);

    Instance::new(
        lib,
        InstanceCreateInfo {
            application_name: Some(config.name.clone()),
            enabled_extensions: all_extensions,
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
    )
    .expect("failed to create Vulkan instance")
}

mod physical_device_selection {
    use std::sync::Arc;
    use vulkano::{
        device::physical::{PhysicalDevice, PhysicalDeviceType},
        instance::Instance,
        memory::MemoryProperties,
    };

    pub fn select_best_physical_device(instance: Arc<Instance>) -> Arc<PhysicalDevice> {
        use std::sync::Arc;

        // Enumerate all physical devices
        let physical_devices: Vec<Arc<PhysicalDevice>> = instance
            .enumerate_physical_devices()
            .expect("Failed to enumerate physical devices")
            .collect();

        physical_devices
            .into_iter()
            .max_by_key(|p| {
                let properties = p.properties();
                let memory_properties = p.memory_properties();

                let total_memory = to_mb(get_physical_device_memory_size(&memory_properties));

                let type_score = match properties.device_type {
                    PhysicalDeviceType::DiscreteGpu => 1000,
                    PhysicalDeviceType::IntegratedGpu => 500,
                    PhysicalDeviceType::VirtualGpu => 100,
                    PhysicalDeviceType::Cpu => 50,
                    _ => 0,
                };
                let memory_score = (total_memory / 100.0) as i32;

                println!(
                    "device: {} (type: {:?}) memory: {} megabytes score: {}",
                    properties.device_name,
                    properties.device_type,
                    total_memory,
                    type_score + memory_score
                );
                type_score + memory_score
            })
            .expect("no suitable physical device found")
    }

    fn get_physical_device_memory_size(memory_properties: &MemoryProperties) -> u64 {
        use std::collections::HashSet;
        let device_local_memory_heaps: HashSet<u32> = memory_properties
            .memory_types
            .iter()
            .filter(|&memory_type| {
                memory_type
                    .property_flags
                    .contains(vulkano::memory::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .map(|memory_type| memory_type.heap_index)
            .collect();

        memory_properties
            .memory_heaps
            .iter()
            .enumerate()
            .filter(|&(i, _)| device_local_memory_heaps.contains(&(i as u32)))
            .map(|(_, heap)| heap.size)
            .sum()
    }

    fn to_mb(bytes: u64) -> f64 {
        bytes as f64 / (1024.0 * 1024.0)
    }
}

mod device_and_queue {
    use std::sync::Arc;
    use vulkano::device::{
        physical::PhysicalDevice, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
        QueueFlags,
    };

    pub fn create_device(
        physical_device: Arc<PhysicalDevice>,
    ) -> (
        Arc<Device>,
        Arc<vulkano::device::Queue>,
        Arc<vulkano::device::Queue>,
    ) {
        let gfx_queue_idx = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(_queue_family_index, queue_family_properties)| {
                queue_family_properties
                    .queue_flags
                    .intersects(QueueFlags::GRAPHICS)
            })
            .expect("failed to obtain a queue with graphics support");

        let compute_queue_idx = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(_queue_family_index, queue_family_properties)| {
                queue_family_properties
                    .queue_flags
                    .intersects(QueueFlags::COMPUTE)
                    && gfx_queue_idx != _queue_family_index
            })
            .unwrap_or(gfx_queue_idx as usize);

        println!(
            "using queue family {} for graphics and queue family {} for compute",
            gfx_queue_idx, compute_queue_idx
        );

        let queue_create_infos = if gfx_queue_idx == compute_queue_idx {
            vec![QueueCreateInfo {
                queue_family_index: gfx_queue_idx as u32,
                ..Default::default()
            }]
        } else {
            vec![
                QueueCreateInfo {
                    queue_family_index: gfx_queue_idx as u32,
                    ..Default::default()
                },
                QueueCreateInfo {
                    queue_family_index: compute_queue_idx as u32,
                    ..Default::default()
                },
            ]
        };

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos,
                enabled_extensions: DeviceExtensions {
                    khr_swapchain: true,
                    ..DeviceExtensions::empty()
                },
                ..Default::default()
            },
        )
        .expect("Failed to create device");

        let gfx_queue = queues.next().unwrap();
        let compute_queue = if gfx_queue_idx == compute_queue_idx {
            gfx_queue.clone()
        } else {
            queues.next().unwrap()
        };

        (device, gfx_queue, compute_queue)
    }
}

pub struct VulkanoConfig {
    pub name: String,
}

pub struct VulkanoContext {
    instance: Arc<Instance>,
    device: Arc<Device>,
    graphics_queue: Arc<Queue>,
    compute_queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
}

impl VulkanoContext {
    pub fn new(config: VulkanoConfig, event_loop: Option<&EventLoop<()>>) -> Self {
        let lib = VulkanLibrary::new().expect("failed to load Vulkan library");
        let instance = create_instance(&config, lib, event_loop);

        let physical_device =
            physical_device_selection::select_best_physical_device(instance.clone());
        let (device, graphics_queue, compute_queue) =
            device_and_queue::create_device(physical_device.clone());

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        Self {
            instance,
            device,
            graphics_queue,
            compute_queue,
            memory_allocator,
        }
    }

    pub fn device_name(&self) -> &str {
        &self.device.physical_device().properties().device_name
    }

    pub fn device_type(&self) -> PhysicalDeviceType {
        self.device.physical_device().properties().device_type
    }

    pub fn max_memory(&self) -> u32 {
        // MARK: this might not be the same as the physical device memory size
        self.device
            .physical_device()
            .properties()
            .max_memory_allocation_count
    }

    pub fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    pub fn graphics_queue(&self) -> &Arc<Queue> {
        &self.graphics_queue
    }

    pub fn compute_queue(&self) -> &Arc<Queue> {
        &self.compute_queue
    }

    pub fn memory_allocator(&self) -> &Arc<StandardMemoryAllocator> {
        &self.memory_allocator
    }
}
