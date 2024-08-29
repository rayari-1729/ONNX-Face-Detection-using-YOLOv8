import sys 
import onnx
import os



if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('please provide model file\nsample usage: python convert.py model.onnx output.onnx')
        sys.exit(0)
    # load the model with onnx, verify it's validity and make the variable input dimension fixed to 640 x 640
    if not os.path.exists(sys.argv[1]):
        print(f"Model file {sys.argv[1]} does not exist")
        sys.exit(1)
    #load and verify the model
    model = onnx.load(sys.argv[1])
    onnx.checker.check_model(model)
    # set the model input dimensions
    model.graph.input[0].type.tensor_type.shape.dim[2].dim_param = '640'
    model.graph.input[0].type.tensor_type.shape.dim[3].dim_param = '640'
    #extract the output directory and file name and save the model
    output_dir = os.path.join(sys.argv[2])
    file_name = ((sys.argv[1]).split('/')[-1]).split('.')[0] + "-fixed.onnx"
    output_path = os.path.join(output_dir, file_name)
    try:
        onnx.save(model, output_path)
        print(f"Saved the model to {output_path}")
    except Exception as e:
        print(f"Error while saving the model: {e}")
        sys.exit(1)