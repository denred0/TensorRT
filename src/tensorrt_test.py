import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import datasets, models
import torchvision.transforms as transforms

from tqdm import tqdm

from onnx import ModelProto
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

import numpy as np
import matplotlib.pyplot as plt
from time import time


# from inceptionv4 import Inceptionv4


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.4)

        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.drop(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        x = self.pool(x)
        x = self.drop(x)

        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.log_softmax(self.fc3(x), dim=1)

        return x


# def imshow(img):


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight)


def train(device, train_loader, val_loader):
    model = Net()
    # model = Inceptionv4()

    model.parameters()

    model.apply(init_weights).to(device)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    no_epochs = 1 + 10

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
            torch.save(model.state_dict(), 'model_cifar.pth')
            val_loss_min = val_loss


def pytorch_test(device, test_loader, b_epoch):
    model_test = Net()
    # model_test = Inceptionv4()
    model_test.load_state_dict(torch.load('model_cifar.pth'))
    model_test = model_test.to(device)
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


def build_engine(TRT_LOGGER, onnx_path, shape=[32, 3, 32, 32], MAX_BATCH=100):
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


def convert_model_to_onnx(onnx_path, engine_name, model, device):
    input_shape = (batch_size, 3, 32, 32)
    inputs = torch.ones(*input_shape)
    inputs = inputs.to(device)
    torch.onnx.export(model, inputs, onnx_path,
                      input_names=None, output_names=None, dynamic_axes=None)


def load_images_to_buffer(pics, pagelocked_buffer):
    preprocessed = np.asarray(pics).ravel()
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
    print('TensorRT version:', trt.__version__)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.RandomRotation(20),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = datasets.CIFAR10(
        root='data', train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR10(
        root='data', train=False, download=True, transform=test_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)

    # train/val split
    val_size = 0.1
    split = int(np.floor((val_size * num_train)))
    train_idx, val_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    batch_size = 100

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    no_classes = len(classes)

    # visualization
    # data_iter = iter(train_loader)
    # images, labels = data_iter.next()
    # images = images.numpy()

    # fig = plt.figure(figsize=(9, 9))
    # fig.suptitle('CIFAR10', fontsize=18)

    # for im in np.arange(batch_size):
    #     fig.tight_layout()  #
    #     ax = fig.add_subplot(10, int(batch_size / 10), im + 1, xticks=[], yticks=[])
    #     ax.set_title(classes[labels[im]], fontsize=8)
    #     # plt.rcParams.update({'axes.titlesize': 'small'})
    #     images[im] = images[im] / 2 + 0.5
    #     plt.imshow(np.transpose(images[im], (1, 2, 0)))

    # plt.show()

    # train
    train(device=device, train_loader=train_loader, val_loader=val_loader)

    # pytorch native inference
    b_epoch = 10
    model_test, taken_pytorch, acc_pytorch = pytorch_test(
        device=device, test_loader=test_loader, b_epoch=b_epoch)

    # Converting model in .pth to .onnx format
    onnx_path = "workspace/model_cifar.onnx"
    engine_name = "workspace/model_fp16.plan"
    convert_model_to_onnx(
        onnx_path=onnx_path, engine_name=engine_name, model=model_test, device=device)

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

    engine = build_engine(TRT_LOGGER=TRT_LOGGER,
                          onnx_path=onnx_path, shape=shape)
    save_engine(engine, engine_name)

    test_acc = 0
    start = time()

    for i in range(b_epoch):
        for image, label in test_loader:
            temp = np.asarray(image, dtype=np.float32)

            h_input, d_input, h_output, d_output, stream = allocate_buffers(
                engine)
            out = do_inference(engine, temp, h_input, d_input,
                               h_output, d_output, stream, batch_size)
            out = torch.from_numpy(out.reshape(batch_size, -1))

            _, pred = torch.max(out, dim=1)
            equal = pred == label.view(*pred.shape)

            test_acc += torch.mean(equal.type(torch.FloatTensor))

    test_acc /= b_epoch
    taken = time() - start

    print("Accuracy PyTorch is: {:.2f}%".format(acc_pytorch))
    print("Time taken PyTorch: {:.2f}s".format(taken_pytorch))
    print()
    print("Accuracy TensorRT is: {:.2f}%".format(test_acc * 100 / len(test_loader)))
    print("Time taken TensorRT: {:.2f}s".format(taken))
