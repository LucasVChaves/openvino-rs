use openvino::prelude::*;

fn main() {
    // Create a new OpenVINO core
    let mut core = Core::new().unwrap();

    // Load the model
    let model = core.read_model("path/to/model.xml").unwrap();

    // Create an inference request
    let mut infer_request = core.create_infer_request(model).unwrap();

    // Set input data
    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    infer_request.set_input("input", input_data).unwrap();

    // Run inference
    infer_request.infer().unwrap();

    // Get output data
    let output_data = infer_request.get_output("output").unwrap();

    // Print output data
    println!("Output: {:?}", output_data);
}