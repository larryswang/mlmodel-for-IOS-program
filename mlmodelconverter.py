import numpy as np
from tensorflow.python.tools.freeze_graph import freeze_graph
import tfcoreml
import linear_model

linear_model.train()

# Provide these to run freeze_graph:
# Graph definition file, stored as protobuf TEXT
graph_def_file = './model.pbtxt'
# Trained model's checkpoint name
checkpoint_file = './checkpoints/model.ckpt'
# Frozen model's output name
frozen_model_file = './frozen_model.pb'
# Output nodes. If there're multiple output ops, use comma separated string, e.g. "out1,out2".
output_node_names = 'Softmax' 


# Call freeze graph
freeze_graph(input_graph=graph_def_file,
             input_saver="",
             input_binary=False,
             input_checkpoint=checkpoint_file,
             output_node_names=output_node_names,
             restore_op_name="save/restore_all",
             filename_tensor_name="save/Const:0",
             output_graph=frozen_model_file,
             clear_devices=True,
             initializer_nodes="")

input_tensor_shapes = {"Placeholder:0":[1,6]} # batch size is 1
# Output CoreML model path
coreml_model_file = './model.mlmodel'
output_tensor_names = ['Softmax:0']


# Call the converter
coreml_model = tfcoreml.convert(
        tf_model_path=frozen_model_file, 
        mlmodel_path=coreml_model_file, 
        input_name_shape_dict=input_tensor_shapes,
        output_feature_names=output_tensor_names)

