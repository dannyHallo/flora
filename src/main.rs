use image::{ImageBuffer, Rgba};
use vulkano::{
    pipeline::Pipeline,
    sync::{self, GpuFuture},
};

// Shader module
mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 460

            layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

            layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;

            void main() {
                vec2 norm_coordinates = (gl_GlobalInvocationID.xy + vec2(0.5)) / vec2(imageSize(img));
                vec2 c = (norm_coordinates - vec2(0.5)) * 2.0 - vec2(1.0, 0.0);

                vec2 z = vec2(0.0, 0.0);
                float i;
                for (i = 0.0; i < 1.0; i += 0.005) {
                    z = vec2(
                        z.x * z.x - z.y * z.y + c.x,
                        z.y * z.x + z.x * z.y + c.y
                    );

                    if (length(z) > 4.0) {
                        break;
                    }
                }

                vec4 to_write = vec4(vec3(i), 1.0);
                imageStore(img, ivec2(gl_GlobalInvocationID.xy), to_write);
            }
        ",
    }
}

// Vulkan Library Loader
mod vulkan_setup {
    use std::sync::Arc;
    use vulkano::VulkanLibrary;

    pub fn load_vulkan_library() -> Arc<VulkanLibrary> {
        VulkanLibrary::new().expect("Failed to load Vulkan library")
    }
}

// Instance Creation
mod instance {
    use std::sync::Arc;
    use vulkano::{
        instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
        VulkanLibrary,
    };

    pub fn create_instance(lib: Arc<VulkanLibrary>) -> Arc<Instance> {
        Instance::new(
            lib,
            InstanceCreateInfo {
                application_name: Some("Flora".into()),
                enabled_extensions: vulkano::instance::InstanceExtensions {
                    khr_portability_enumeration: true,
                    ..vulkano::instance::InstanceExtensions::empty()
                },
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                ..Default::default()
            },
        )
        .expect("Failed to create Vulkan instance")
    }
}

mod physical_device_selection {
    use std::sync::Arc;
    use vulkano::{
        device::physical::{PhysicalDevice, PhysicalDeviceType},
        instance::Instance,
        memory::MemoryProperties,
    };

    pub fn get_physical_device_memory_size(memory_properties: &MemoryProperties) -> u64 {
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
                    "Device: {} (type: {:?}) Memory: {}MB Score: {}",
                    properties.device_name,
                    properties.device_type,
                    total_memory,
                    type_score + memory_score
                );
                type_score + memory_score
            })
            .expect("No suitable physical device found")
    }

    fn to_mb(bytes: u64) -> f64 {
        bytes as f64 / (1024.0 * 1024.0)
    }
}

// Logical Device and Queue Creation
mod device_and_queue {
    use std::sync::Arc;
    use vulkano::device::{
        physical::PhysicalDevice, Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags,
    };

    pub fn create_device_and_queue(
        physical_device: Arc<PhysicalDevice>,
    ) -> (Arc<Device>, Arc<vulkano::device::Queue>) {
        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(_queue_family_index, queue_family_properties)| {
                queue_family_properties
                    .queue_flags
                    .contains(QueueFlags::GRAPHICS)
            })
            .expect("couldn't find a graphical queue family")
            as u32;

        // Create the logical device and the queue
        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .expect("Failed to create device");

        let queue = queues.next().unwrap();
        (device, queue)
    }
}

// Memory Allocator Setup
mod memory {
    use std::sync::Arc;
    use vulkano::{device::Device, memory::allocator::StandardMemoryAllocator};

    pub fn create_standard_memory_allocator(device: Arc<Device>) -> Arc<StandardMemoryAllocator> {
        Arc::new(StandardMemoryAllocator::new_default(device))
    }
}

// Buffer Creation
mod buffer {
    use std::sync::Arc;
    use vulkano::{
        buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
        memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    };

    pub fn create_image_buffer(memory_allocator: &Arc<StandardMemoryAllocator>) -> Subbuffer<[u8]> {
        Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            (0..1024 * 1024 * 4).map(|_| 0u8),
        )
        .expect("failed to create buffer")
    }
}

// Shader and Pipeline Setup
mod pipeline_setup {
    use std::sync::Arc;
    use vulkano::device::Device;
    use vulkano::pipeline::compute::ComputePipelineCreateInfo;
    use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
    use vulkano::pipeline::{ComputePipeline, PipelineLayout, PipelineShaderStageCreateInfo};

    use crate::cs;

    pub fn create_compute_pipeline(device: &Arc<Device>) -> Arc<ComputePipeline> {
        let shader = cs::load(device.clone()).expect("Failed to create shader module");
        let cs = shader.entry_point("main").unwrap();

        let stage = PipelineShaderStageCreateInfo::new(cs);
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .expect("failed to create compute pipeline")
    }
}

mod descriptor_set_util {
    use std::sync::Arc;
    use vulkano::{
        descriptor_set::{
            allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
        },
        device::Device,
        image::{view::ImageView, Image},
        pipeline::PipelineLayout,
    };

    fn get_descriptor_set_allocator(device: &Arc<Device>) -> StandardDescriptorSetAllocator {
        StandardDescriptorSetAllocator::new(device.clone(), Default::default())
    }

    pub fn create_descriptor_set(
        device: &Arc<Device>,
        pipeline_layout: &Arc<PipelineLayout>,
        image: Arc<Image>,
    ) -> Arc<PersistentDescriptorSet> {
        let descriptor_set_allocator = get_descriptor_set_allocator(device);

        let view = ImageView::new_default(image.clone()).expect("Failed to create image view");

        let set_idx = 0;
        let descriptor_set_layout = pipeline_layout
            .set_layouts()
            .get(set_idx)
            .expect("Failed to get descriptor set layout");

        PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            descriptor_set_layout.clone(), // MMM test this
            [WriteDescriptorSet::image_view(0, view.clone())], // 0 is the binding
            [],
        )
        .expect("Failed to create descriptor set")
    }
}

// Command Buffer Recording and Execution
mod commands {
    use std::sync::Arc;
    use vulkano::{
        buffer::Subbuffer,
        command_buffer::{
            allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
            AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo,
            PrimaryAutoCommandBuffer,
        },
        descriptor_set::PersistentDescriptorSet,
        device::{Device, Queue},
        image::Image,
        pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    };

    pub fn create_command_buffer_allocator(device: &Arc<Device>) -> StandardCommandBufferAllocator {
        StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        )
    }

    pub fn create_command_buffer(
        command_buffer_allocator: &StandardCommandBufferAllocator,
        queue: &Arc<Queue>,
        compute_pipeline: &Arc<ComputePipeline>,
        descriptor_set: &Arc<PersistentDescriptorSet>,
        image: &Arc<Image>,
        buffer: &Subbuffer<[u8]>,
    ) -> Arc<PrimaryAutoCommandBuffer> {
        let mut builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("Failed to create command buffer builder");

        builder
            .bind_pipeline_compute(compute_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                compute_pipeline.layout().clone(),
                0,
                descriptor_set.clone(),
            )
            .unwrap()
            .dispatch([1024 / 8, 1024 / 8, 1])
            .unwrap()
            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                image.clone(),
                buffer.clone(),
            ))
            .unwrap();

        builder.build().unwrap()
    }
}

mod image_util {
    use std::sync::Arc;
    use vulkano::{
        image::Image,
        memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    };

    pub fn create_image(memory_allocator: &Arc<StandardMemoryAllocator>) -> Arc<Image> {
        use vulkano::format::Format;
        use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};

        Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_UNORM,
                extent: [1024, 1024, 1],
                usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .expect("Failed to create image")
    }
}

fn main() {
    let lib = vulkan_setup::load_vulkan_library();
    let instance = instance::create_instance(lib);
    let physical_device = physical_device_selection::select_best_physical_device(instance.clone());
    let (device, queue) = device_and_queue::create_device_and_queue(physical_device.clone());

    let memory_allocator = memory::create_standard_memory_allocator(device.clone());
    let command_buffer_allocator = commands::create_command_buffer_allocator(&device);

    let image = image_util::create_image(&memory_allocator);
    let image_buffer = buffer::create_image_buffer(&memory_allocator);

    let compute_pipeline = pipeline_setup::create_compute_pipeline(&device);
    let descriptor_set = descriptor_set_util::create_descriptor_set(
        &device,
        &compute_pipeline.layout(),
        image.clone(),
    );

    let command_buffer = commands::create_command_buffer(
        &command_buffer_allocator,
        &queue,
        &compute_pipeline,
        &descriptor_set,
        &image,
        &image_buffer,
    );

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .expect("Failed to execute command buffer")
        .then_signal_fence_and_flush()
        .expect("Failed to signal fence and flush");

    future.wait(None).expect("Failed to wait for fence");

    let buffer_content = image_buffer.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
    image.save("image.png").unwrap();
    println!("Everything succeeded!");
}
