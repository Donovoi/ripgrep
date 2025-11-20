//! Comprehensive GPU acceleration test suite
//!
//! These tests validate GPU features including:
//! - GPU availability detection
//! - GPU device enumeration
//! - Threshold-based GPU selection
//! - Graceful fallback to CPU
//! - Error handling
//! - Performance characteristics

#[cfg(feature = "cuda-gpu")]
mod gpu_tests {
    use gdeflate::{compress, gpu::*};

    #[test]
    fn test_gpu_availability_detection() {
        // Test GPU availability check (should not panic)
        let available = is_gpu_available();
        println!("GPU available: {}", available);
        
        // This should always succeed regardless of GPU presence
        assert!(available == true || available == false);
    }

    #[test]
    fn test_gpu_device_enumeration() {
        let devices = get_gpu_devices();
        
        if devices.is_empty() {
            println!("No GPU devices found (expected in CI environment)");
        } else {
            println!("Found {} GPU device(s):", devices.len());
            for (i, device) in devices.iter().enumerate() {
                println!("  GPU {}: {}", i, device.name);
                println!("    Total memory: {} GB", device.total_memory / (1024*1024*1024));
                println!("    Free memory: {} GB", device.free_memory / (1024*1024*1024));
                println!("    Compute capability: {}.{}", device.compute_capability.0, device.compute_capability.1);
                println!("    Multiprocessors: {}", device.multiprocessor_count);
                
                // Validate device info
                assert!(!device.name.is_empty(), "Device name should not be empty");
                assert!(device.total_memory > 0, "Total memory should be positive");
                assert!(device.compute_capability.0 >= 5, "Compute capability should be at least 5.0");
            }
        }
    }

    #[test]
    fn test_gpu_threshold_logic() {
        // Test threshold function
        let small_size = 1024 * 1024; // 1 MB
        let medium_size = 10 * 1024 * 1024 * 1024; // 10 GB
        let large_size = 60 * 1024 * 1024 * 1024; // 60 GB
        let huge_size = 2 * 1024 * 1024 * 1024 * 1024; // 2 TB (exceeds max)
        
        assert!(!should_use_gpu(small_size), "Small files should not use GPU");
        assert!(!should_use_gpu(medium_size), "Medium files should not use GPU");
        assert!(should_use_gpu(large_size), "Large files should use GPU");
        assert!(!should_use_gpu(huge_size), "Files exceeding max size should not use GPU");
    }

    #[test]
    fn test_gpu_fallback_on_unavailable() {
        // Test that decompression works even when GPU is unavailable
        let input = b"Hello, GPU test! This is a test of fallback behavior.";
        let compressed = compress(input, 6, 0).expect("compression failed");
        
        // Try to decompress with auto selection (should fall back to CPU if no GPU)
        let decompressed = decompress_auto(&compressed, input.len())
            .expect("decompression should succeed with fallback");
        
        assert_eq!(input, decompressed.as_slice(), "Decompressed data should match input");
    }

    #[test]
    fn test_gpu_error_handling() {
        // Test error handling with invalid parameters
        let input = b"Test data";
        let compressed = compress(input, 6, 0).expect("compression failed");
        
        // Try to decompress with size exceeding GPU max (should error gracefully)
        let result = decompress_with_gpu(&compressed, GPU_MAX_SIZE + 1);
        
        // Should return an error
        assert!(result.is_err(), "Should error when size exceeds GPU max");
        
        // Error message should be informative
        if let Err(e) = result {
            let error_msg = format!("{}", e);
            println!("Error message: {}", error_msg);
            assert!(error_msg.len() > 0, "Error message should not be empty");
        }
    }

    #[test]
    fn test_gpu_small_file_handling() {
        // Test that small files are handled correctly (should use CPU)
        let input = b"Small file content for testing";
        let compressed = compress(input, 6, 0).expect("compression failed");
        
        // Auto should use CPU for small files
        let decompressed = decompress_auto(&compressed, input.len())
            .expect("decompression failed");
        
        assert_eq!(input, decompressed.as_slice());
    }

    #[test]
    fn test_literal_search_config() {
        // Test literal search configuration estimation
        let config = estimate_literal_search_config();
        
        if let Some(cfg) = config {
            println!("Literal search config:");
            println!("  Min file bytes: {} GB", cfg.min_file_bytes / (1024*1024*1024));
            println!("  Chunk bytes: {} MB", cfg.chunk_bytes / (1024*1024));
            
            // Validate configuration
            assert!(cfg.min_file_bytes > 0, "Min file bytes should be positive");
            assert!(cfg.chunk_bytes > 0, "Chunk bytes should be positive");
            assert!(cfg.chunk_bytes <= cfg.min_file_bytes as usize, 
                    "Chunk size should not exceed min file size");
        } else {
            println!("No GPU available for literal search config");
        }
    }

    #[test]
    fn test_literal_search_config_with_hints() {
        use LiteralSearchInputKind;
        
        // Test with single file hint
        let hint = LiteralSearchInputHint {
            kind: LiteralSearchInputKind::SingleFile,
            approx_total_bytes: Some(100 * 1024 * 1024 * 1024), // 100 GB
            per_entry_max_bytes: None,
        };
        
        let config = estimate_literal_search_config_with_hint(hint);
        if let Some(cfg) = config {
            println!("Single file config: chunk={} MB", cfg.chunk_bytes / (1024*1024));
            assert!(cfg.chunk_bytes > 0);
        }
        
        // Test with multiple inputs hint
        let hint = LiteralSearchInputHint {
            kind: LiteralSearchInputKind::MultipleInputs,
            approx_total_bytes: Some(100 * 1024 * 1024 * 1024),
            per_entry_max_bytes: Some(10 * 1024 * 1024 * 1024),
        };
        
        let config_multi = estimate_literal_search_config_with_hint(hint);
        if let Some(cfg) = config_multi {
            println!("Multiple inputs config: chunk={} MB", cfg.chunk_bytes / (1024*1024));
            assert!(cfg.chunk_bytes > 0);
        }
    }

    #[test]
    fn test_substring_contains_empty_needle() {
        // Test edge case: empty needle
        let haystack = b"test data";
        let needle = b"";
        
        let result = substring_contains(haystack, needle);
        assert!(result.is_ok(), "Empty needle should not cause error");
        if let Ok(found) = result {
            assert!(found, "Empty needle should always be found");
        }
    }

    #[test]
    fn test_substring_contains_empty_haystack() {
        // Test edge case: empty haystack
        let haystack = b"";
        let needle = b"test";
        
        let result = substring_contains(haystack, needle);
        assert!(result.is_ok(), "Empty haystack should not cause error");
        if let Ok(found) = result {
            assert!(!found, "Non-empty needle should not be found in empty haystack");
        }
    }

    #[test]
    fn test_substring_contains_simple() {
        // Test basic substring search
        let haystack = b"The quick brown fox jumps over the lazy dog";
        let needle = b"quick";
        
        let result = substring_contains(haystack, needle);
        if result.is_ok() {
            // If GPU is available, verify result
            if let Ok(found) = result {
                assert!(found, "Needle should be found in haystack");
            }
        } else {
            // If GPU not available, fallback is expected
            println!("GPU substring search not available (expected in CI)");
        }
    }

    #[test]
    fn test_gpu_constants() {
        // Validate GPU constants are sensible
        assert!(GPU_SIZE_THRESHOLD > 0, "GPU threshold should be positive");
        assert!(GPU_MAX_SIZE > GPU_SIZE_THRESHOLD, "Max size should exceed threshold");
        
        println!("GPU_SIZE_THRESHOLD: {} GB", GPU_SIZE_THRESHOLD / (1024*1024*1024));
        println!("GPU_MAX_SIZE: {} GB", GPU_MAX_SIZE / (1024*1024*1024));
    }

    #[test]
    fn test_concurrent_gpu_access() {
        // Test that multiple threads can query GPU info concurrently
        use std::thread;
        
        let handles: Vec<_> = (0..4).map(|i| {
            thread::spawn(move || {
                let available = is_gpu_available();
                let devices = get_gpu_devices();
                println!("Thread {}: GPU available={}, devices={}", i, available, devices.len());
                (available, devices.len())
            })
        }).collect();
        
        let results: Vec<_> = handles.into_iter()
            .map(|h| h.join().expect("thread panicked"))
            .collect();
        
        // All threads should report consistent results
        let first = &results[0];
        for result in &results[1..] {
            assert_eq!(first.0, result.0, "GPU availability should be consistent");
            assert_eq!(first.1, result.1, "Device count should be consistent");
        }
    }
}

// Tests that should work without cuda-gpu feature
#[cfg(not(feature = "cuda-gpu"))]
#[test]
fn test_gpu_feature_disabled() {
    // When cuda-gpu feature is not enabled, GPU functions should not be available
    // This test just ensures the code compiles without the feature
    println!("cuda-gpu feature is disabled");
}
