import numpy as np
import pickle
import torch
import timm

# pytorch related imports
from torch.utils.data import DataLoader
from torch import nn

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.dataset import ICPDataset

# import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2

from time import time
from pathlib import Path

from onnx import ModelProto
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

albumentations_transform = A.Compose([
    A.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
    ToTensorV2()
])


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_dir = 'data_inceptionv4/data_simpsons'

    path = Path(data_dir)

    train_val_files = list(path.rglob('*.jpg'))
    train_val_labels = [path.parent.name for path in train_val_files]

    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(train_val_labels)
    num_classes = len(np.unique(encoded))

    # save labels dict to file
    with open('label_encoder.pkl', 'wb') as le_dump_file:
        pickle.dump(label_encoder, le_dump_file)

    train_files, val_test_files = train_test_split(train_val_files, test_size=0.3, stratify=train_val_labels)

    train_labels = [path.parent.name for path in train_files]
    train_labels = label_encoder.transform(train_labels)
    train_data = train_files, train_labels

    # without test step
    # val_labels = [path.parent.name for path in val_test_files]
    # val_labels = label_encoder.transform(val_labels)
    # val_data = val_test_files, val_labels

    # with test step
    val_test_labels = [path.parent.name for path in val_test_files]
    val_files, test_files = train_test_split(val_test_files, test_size=0.5, stratify=val_test_labels)

    val_labels = [path.parent.name for path in val_files]
    val_labels = label_encoder.transform(val_labels)

    test_labels = [path.parent.name for path in test_files]
    test_labels = label_encoder.transform(test_labels)

    val_data = val_files, val_labels
    test_data = test_files, test_labels

    train_dataset = ICPDataset(data=train_data, input_resize=299, preprocessing=albumentations_transform)
    val_dadaset = ICPDataset(data=val_data, input_resize=299, preprocessing=albumentations_transform)
    test_dataset = ICPDataset(data=test_data, input_resize=299, preprocessing=albumentations_transform)

    batch_size = 10
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_dadaset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    model_type = 'inception_v4'
    b_epoch = 10
    train(model_type, num_classes, device, train_loader, val_loader, no_epochs=b_epoch)

    model_test, taken_pytorch, acc_pytorch = pytorch_test(model_type=model_type, num_classes=num_classes,
                                                          device=device, test_loader=test_loader, b_epoch=b_epoch)

    onnx_path = "workspace_inceptionv4/model_inceptionv4.onnx"
    engine_name = "workspace_inceptionv4/model_fp16.plan"
    convert_model_to_onnx(
        onnx_path=onnx_path, engine_name=engine_name, model=model_test, device=device, batch_size=batch_size)

    # TensorRT flow
    verbose = True
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING) if verbose else trt.Logger()
    trt_runtime = trt.Runtime(TRT_LOGGER)

    model = ModelProto()
    with open(onnx_path, "rb") as f:
        model.ParseFromString(f.read())

    d0 = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    d1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
    d2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value
    shape = [batch_size, d0, d1, d2]

    print('shape', shape)

    builder = trt.Builder(TRT_LOGGER)
    # print('builder', builder)

    engine = build_engine(TRT_LOGGER=TRT_LOGGER, onnx_path=onnx_path, shape=shape)
    save_engine(engine, engine_name)

    test_acc = 0
    start = time()

    for i in range(b_epoch):
        for image, label in test_loader:
            temp = np.asarray(image, dtype=np.float32)

            h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)

            # last batch
            out_size = batch_size
            if label.shape[0] < batch_size:
                out_size = label.shape[0]

            out = do_inference(engine, temp, h_input, d_input,
                               h_output, d_output, stream, out_size)
            out = torch.from_numpy(out.reshape(out_size, -1))

            _, pred = torch.max(out, dim=1)
            equal = pred == label.view(*pred.shape)

            test_acc += torch.mean(equal.type(torch.FloatTensor))

    test_acc /= b_epoch
    taken = time() - start

    print("\nAccuracy PyTorch is: {:.2f}%".format(acc_pytorch))
    print("Time taken PyTorch: {:.2f}s".format(taken_pytorch))
    print("\nAccuracy TensorRT is: {:.2f}%".format(test_acc * 100 / len(test_loader)))
    print("Time taken TensorRT: {:.2f}s".format(taken))


def train(model_type, num_classes, device, train_loader, val_loader, no_epochs):
    model = timm.create_model(model_type, pretrained=True)
    in_features = model.last_linear.in_features
    # in_features = model.fc.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    model.to(device)

    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # no_epochs = 1 + 10

    val_loss_min = np.Inf

    for epoch in range(1, no_epochs):

        start = time()

        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        model.train()
        for idx, (data, target) in tqdm(enumerate(train_loader)):
            data, target = data.to(device), target.to(device)

            output = model(data)

            loss = criterion(output, target)

            optimizer.zero_grad()

            loss.backward()

            # with amp.scale_loss(loss,optimizer) as scaled_loss:
            #    scaled_loss.backward()

            optimizer.step()

            train_loss += loss.item()

            _, pred = torch.max(output, dim=1)

            equals = pred == target.view(*pred.shape)

            train_acc += torch.mean(equals.type(torch.FloatTensor))

        model.eval()
        for idx, (data, target) in tqdm(enumerate(val_loader)):
            data, target = data.to(device), target.to(device)

            output = model(data)

            loss = criterion(output, target)

            val_loss += loss.item()

            _, pred = torch.max(output, dim=1)

            equals = pred == target.view(*pred.shape)

            val_acc += torch.mean(equals.type(torch.FloatTensor))

        train_loss = train_loss / len(train_loader)
        train_acc = train_acc * 100 / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_acc = val_acc * 100 / len(val_loader)

        end = time()
        taken = end - start

        print(
            'Epoch: {} \tTime: {:.3f} \nTraining Loss: {:.6f} \tTraining Acc: {:.2f} \tValidation Loss: {:.6f} \tValidation Acc: {:.2f}'.format(
                epoch, taken, train_loss, train_acc, val_loss, val_acc))

        if val_loss <= val_loss_min:
            print(
                'Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(val_loss_min, val_loss))
            torch.save(model.state_dict(), 'model_inceptionv4.pth')
            val_loss_min = val_loss


def pytorch_test(model_type, num_classes, device, test_loader, b_epoch):
    model_test = timm.create_model(model_type, pretrained=True)
    # in_features = model_test.fc.in_features
    in_features = model_test.last_linear.in_features
    model_test.classifier = nn.Linear(in_features, num_classes)
    model_test.load_state_dict(torch.load('model_inceptionv4.pth'))
    model_test.to(device)
    model_test.eval()

    test_acc = 0
    start = time()
    for i in range(b_epoch):

        for idx, (data, target) in tqdm(enumerate(test_loader)):
            data, target = data.to(device), target.to(device)

            # optimizer.zero_grad()

            output = model_test(data)

            _, pred = torch.max(output, dim=1)

            equal = pred == target.view(*pred.shape)

            test_acc += torch.mean(equal.type(torch.FloatTensor))

    test_acc /= b_epoch
    taken = time() - start
    print("Accuracy is: {:.2f}%".format(test_acc * 100 / len(test_loader)))
    print("Time taken: {:.2f}s".format(taken))

    acc = test_acc * 100 / len(test_loader)

    return model_test, taken, acc


def build_engine(TRT_LOGGER, onnx_path, shape, MAX_BATCH=100):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config() as config, builder.create_network(
            1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.fp16_mode = True
        builder.max_workspace_size = 1 << 30  # 1GB
        builder.max_batch_size = MAX_BATCH


        profile = builder.create_optimization_profile()
        config.max_workspace_size = (3072 << 20)
        config.add_optimization_profile(profile)
        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())
        network.get_input(0).shape = shape
        engine = builder.build_cuda_engine(network)
        return engine


def save_engine(engine, file_name):
    buf = engine.serialize()
    with open(file_name, 'wb') as f:
        f.write(buf)


def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


def convert_model_to_onnx(onnx_path, engine_name, model, device, batch_size):
    input_shape = (batch_size, 3, 299, 299)
    inputs = torch.ones(*input_shape)
    inputs = inputs.to(device)
    torch.onnx.export(model, inputs, onnx_path,
                      input_names=None, output_names=None, dynamic_axes=None)


def load_images_to_buffer(pics, pagelocked_buffer):
    preprocessed = np.asarray(pics).ravel()

    # last batch. Check if count of images is less than batch_size
    if pagelocked_buffer.size > preprocessed.size:
        pagelocked_buffer = np.zeros(preprocessed.size)

    np.copyto(pagelocked_buffer, preprocessed)


def do_inference(engine, pics_1, h_input_1, d_input_1, h_output, d_output, stream, batch_size):
    load_images_to_buffer(pics_1, h_input_1)

    with engine.create_execution_context() as context:
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input_1, h_input_1, stream)

        # Run inference.
        context.profiler = trt.Profiler()
        context.execute(batch_size=batch_size, bindings=[
            int(d_input_1), int(d_output)])

        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()
        # Return the host output.
        out = (h_output)
        return out


def allocate_buffers(engine):
    # host cpu mem
    h_in_size = trt.volume(engine.get_binding_shape(0))
    h_out_size = trt.volume(engine.get_binding_shape(1))

    h_in_dtype = trt.nptype(engine.get_binding_dtype(0))
    h_out_dtype = trt.nptype(engine.get_binding_dtype(1))

    h_input = cuda.pagelocked_empty(h_in_size, h_in_dtype)
    h_output = cuda.pagelocked_empty(h_out_size, h_out_dtype)

    # allocate gpu mem
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()

    return h_input, d_input, h_output, d_output, stream


if __name__ == '__main__':
    main()
