import onnxruntime as rt
import numpy as np
import time

def check_available_providers():
    """Check and print available execution providers."""
    available_providers = rt.get_available_providers()
    print("Available ONNX Runtime execution providers:")
    for provider in available_providers:
        print(f"  - {provider}")

    return available_providers

def test_provider_availability():
    """Test if specific providers are available by creating a minimal session."""
    available_providers = rt.get_available_providers()

    # Create a minimal dummy model in memory using numpy arrays
    # We'll use a simple identity model for testing
    try:
        import onnx
        from onnx import helper, TensorProto

        # Create a simple identity model with compatible IR version
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])
        output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 224, 224])

        identity_node = helper.make_node(
            'Identity',
            inputs=['input'],
            outputs=['output']
        )

        graph = helper.make_graph(
            [identity_node],
            'identity_test_model',
            [input_tensor],
            [output_tensor]
        )

        # Create model with explicit IR version and opset compatible with ONNXRuntime
        model = helper.make_model(graph, producer_name='identity-test-model')
        model.ir_version = 7  # Set to a lower IR version for compatibility
        # Clear any default opset imports and set to a known compatible version
        del model.opset_import[:]
        model.opset_import.extend([helper.make_opsetid("", 11)])  # Use opset 11 for compatibility

        # Test CPU provider
        if 'CPUExecutionProvider' in available_providers:
            print("\nTesting CPU execution provider...")
            try:
                sess = rt.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
                input_name = sess.get_inputs()[0].name
                output_name = sess.get_outputs()[0].name
                dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)

                start_time = time.time()
                _ = sess.run([output_name], {input_name: dummy_input})
                end_time = time.time()

                print(f"CPU inference completed in {end_time - start_time:.4f} seconds")
                print(f"Using provider: {sess.get_providers()[0]}")
            except Exception as e:
                print(f"Failed to run with CPU provider: {e}")

        # Test CUDA provider if available
        if 'CUDAExecutionProvider' in available_providers:
            print("\nTesting CUDA execution provider...")
            try:
                sess = rt.InferenceSession(model.SerializeToString(), providers=['CUDAExecutionProvider'])
                input_name = sess.get_inputs()[0].name
                output_name = sess.get_outputs()[0].name
                dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)

                start_time = time.time()
                _ = sess.run([output_name], {input_name: dummy_input})
                end_time = time.time()

                print(f"CUDA inference completed in {end_time - start_time:.4f} seconds")
                print(f"Using provider: {sess.get_providers()[0]}")
            except Exception as e:
                print(f"Failed to run with CUDA provider: {e}")

    except ImportError:
        print("\nONNX not available, skipping detailed provider tests")
        print("However, we can still check provider availability:")

        if 'CPUExecutionProvider' in available_providers:
            print("  - CPUExecutionProvider: Available")
        else:
            print("  - CPUExecutionProvider: Not available")

        if 'CUDAExecutionProvider' in available_providers:
            print("  - CUDAExecutionProvider: Available")
        else:
            print("  - CUDAExecutionProvider: Not available")

        if 'TensorrtExecutionProvider' in available_providers:
            print("  - TensorrtExecutionProvider: Available")
        else:
            print("  - TensorrtExecutionProvider: Not available")


# --- Main execution ---
print("ONNX Runtime version:", rt.__version__)
available_providers = check_available_providers()

# Test provider availability
test_provider_availability()
