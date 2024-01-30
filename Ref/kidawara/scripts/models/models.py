#! -*- coding: utf-8
import typing
from collections import OrderedDict

import torch
import torch.nn as nn

__all__ = ["ConvModel", "ConvModelLinear"]


class ConvModel(nn.Module):
    def __init__(self, in_shape: int = (6, 65, 8),
                 events=["ROLL", "RUN", "DOOR"],
                 common_channels=[32, 64],
                 common_kernel_size=(8, 1),
                 common_strides=(3, 1),
                 event_channels=[64],
                 event_kernel_size=(1, 8),
                 event_strides=(1, 1),
                 event_features=[128, 32]):
        super(ConvModel, self).__init__()
        self.common_layers, out_channel, out_shape = self.__build_common_layers__([in_shape[0]] + [c for c in common_channels if c > 0],
                                                                                  kernel_size=common_kernel_size,
                                                                                  stride=common_strides,
                                                                                  in_shape=in_shape[1:])
        self.event_layers = OrderedDict()
        for event in events:
            layer, _ = self.__build_event_layers__(event,
                                                   channels=[out_channel]
                                                   + [c for c in event_channels if c > 0],
                                                   kernel_size=event_kernel_size,
                                                   stride=event_strides,
                                                   features=event_features,
                                                   in_shape=out_shape)
            setattr(self, event, layer)  # parameterに登録するするためにattribute化
            self.event_layers[event] = layer

    def events(self):
        return list(self.event_layers.keys())

    def forward(self, x):
        # print("a", x.shape)
        x = self.common_layers(x)
        # print("conv", x.shape)
        probs = OrderedDict()
        for event, layers in self.event_layers.items():
            probs[event] = layers(x)
        return probs

    def __out_shape__(self, in_shape, kernel_size, stride, padding):
        out_shape = tuple((v - k + s + 2*p)//s
                          for v, k, s, p in zip(in_shape, kernel_size, stride, padding))
        return out_shape

    def __build_common_layers__(self,
                                channels,
                                kernel_size=(8, 1),
                                stride=(4, 1),
                                padding=(0, 0),
                                in_shape=(65, 8)):
        layers = OrderedDict()
        assert len(channels) > 1
        assert len(in_shape) == 2
        in_channel, out_shape = channels[0], in_shape

        for i, out_channel in enumerate(channels[1:]):
            # TODO: kernel_size, stride, padding
            if i > 0:
                # layers["batch_norm_%d" % (i-1)] = nn.BatchNorm2d(in_channel)
                layers["relu_%d" % (i-1)] = nn.ReLU()
                layers["dropout_%d" % (i-1)] = nn.Dropout2d()

            layers["conv2d_%d" % i] = nn.Conv2d(in_channel, out_channel, kernel_size,
                                                stride=stride,
                                                padding=padding)  # paddingはデータの両端に実施される
            in_channel = out_channel
            out_shape = self.__out_shape__(
                out_shape, kernel_size, stride, padding)
        return nn.Sequential(layers), out_channel, out_shape

    def __build_event_layers__(self,
                               event: str,
                               channels=[64],
                               kernel_size=(1, 8),
                               stride=(1, 2),
                               padding=(0, 0),
                               features=[28, 32],
                               in_shape=(5, 8)):
        layers = OrderedDict()
        in_channel = channels[0]
        out_shape = in_shape
        for i, out_channel in enumerate(channels[1:]):
            # TODO: kernel_size, stride, padding
            if i > 0:
                # layers["%s_conv2d_batch_norm_%d" %
                #     (event, i-1)] = nn.BatchNorm2d(in_channel)
                layers["%s_conv2d_relu_%d" % (event, i-1)] = nn.ReLU()
                layers["%s_conv2d_dropout_%d" % (event, i-1)] = nn.Dropout2d()
            layers["%s_conv2d_%d" % (event, i)] = nn.Conv2d(in_channel, out_channel, kernel_size,
                                                            stride=stride,
                                                            padding=padding)  # paddingはデータの両端に実施される
            in_channel = out_channel
            out_shape = self.__out_shape__(
                out_shape, kernel_size, stride, padding)
        layers["%s_flatten" % (event)] = torch.nn.Flatten()
        in_feature = 1
        # print(in_channel, out_shape)
        for f in [in_channel] + list(out_shape):
            in_feature *= f
        for i, out_feature in enumerate(features):
            # print("%s_linear_%d" % (event, i), in_feature, out_feature)
            layers["%s_linear_%d" % (event, i)] = nn.Linear(in_feature,
                                                            out_feature)
            # layers["%s_linear_batch_norm_%d" %
            #        (event, i)] = nn.BatchNorm1d(out_feature)
            layers["%s_linear_relu_%d" % (event, i)] = nn.ReLU()
            layers["%s_linear_dropout_%d" % (event, i)] = nn.Dropout()
            in_feature = out_feature
        layers["%s_out" % event] = nn.Linear(out_feature, 1)
        # layers[event] = nn.Softmax(dim=-1)  # to probability # Unit1の場合、すべて1になる
        layers[event] = nn.Sigmoid()  # to 0-1.0 values

        return nn.Sequential(layers), event

class ConvModelLinear(nn.Module):
    def __init__(self, in_shape: int = (6, 65, 8),
                 events=["ROLL", "RUN", "DOOR"],
                 common_channels=[32, 64],
                 common_kernel_size=(8, 1),
                 common_strides=(3, 1),
                 event_channels=[64],
                 event_kernel_size=(1, 8),
                 event_strides=(1, 1),
                 event_features=[128, 32]):
        super(ConvModelLinear, self).__init__()
        self.common_layers, out_channel, out_shape = self.__build_common_layers__([in_shape[0]] + [c for c in common_channels if c > 0],
                                                                                  kernel_size=common_kernel_size,
                                                                                  stride=common_strides,
                                                                                  in_shape=in_shape[1:])
        self.event_layers = OrderedDict()
        for event in events:
            layer, _ = self.__build_event_layers__(event,
                                                   channels=[out_channel]
                                                   + [c for c in event_channels if c > 0],
                                                   kernel_size=event_kernel_size,
                                                   stride=event_strides,
                                                   features=event_features,
                                                   in_shape=out_shape)
            setattr(self, event, layer)  # parameterに登録するするためにattribute化
            self.event_layers[event] = layer

    def events(self):
        return list(self.event_layers.keys())

    def forward(self, x):
        # print("a", x.shape)
        x = self.common_layers(x)
        # print("conv", x.shape)
        probs = OrderedDict()
        for event, layers in self.event_layers.items():
            probs[event] = layers(x)
        return probs

    def __out_shape__(self, in_shape, kernel_size, stride, padding):
        out_shape = tuple((v - k + s + 2*p)//s
                          for v, k, s, p in zip(in_shape, kernel_size, stride, padding))
        return out_shape

    def __build_common_layers__(self,
                                channels,
                                kernel_size=(8, 1),
                                stride=(4, 1),
                                padding=(0, 0),
                                in_shape=(65, 8)):
        layers = OrderedDict()
        assert len(channels) > 1
        assert len(in_shape) == 2
        in_channel, out_shape = channels[0], in_shape

        for i, out_channel in enumerate(channels[1:]):
            # TODO: kernel_size, stride, padding
            if i > 0:
                # layers["batch_norm_%d" % (i-1)] = nn.BatchNorm2d(in_channel)
                layers["relu_%d" % (i-1)] = nn.ReLU()
                layers["dropout_%d" % (i-1)] = nn.Dropout2d()

            layers["conv2d_%d" % i] = nn.Conv2d(in_channel, out_channel, kernel_size,
                                                stride=stride,
                                                padding=padding)  # paddingはデータの両端に実施される
            in_channel = out_channel
            out_shape = self.__out_shape__(
                out_shape, kernel_size, stride, padding)
        return nn.Sequential(layers), out_channel, out_shape

    def __build_event_layers__(self,
                               event: str,
                               channels=[64],
                               kernel_size=(1, 8),
                               stride=(1, 2),
                               padding=(0, 0),
                               features=[28, 32],
                               in_shape=(5, 8)):
        layers = OrderedDict()
        in_channel = channels[0]
        out_shape = in_shape
        for i, out_channel in enumerate(channels[1:]):
            # TODO: kernel_size, stride, padding
            if i > 0:
                # layers["%s_conv2d_batch_norm_%d" %
                #     (event, i-1)] = nn.BatchNorm2d(in_channel)
                layers["%s_conv2d_relu_%d" % (event, i-1)] = nn.ReLU()
                layers["%s_conv2d_dropout_%d" % (event, i-1)] = nn.Dropout2d()
            layers["%s_conv2d_%d" % (event, i)] = nn.Conv2d(in_channel, out_channel, kernel_size,
                                                            stride=stride,
                                                            padding=padding)  # paddingはデータの両端に実施される
            in_channel = out_channel
            out_shape = self.__out_shape__(
                out_shape, kernel_size, stride, padding)
        layers["%s_flatten" % (event)] = torch.nn.Flatten()
        in_feature = 1
        # print(in_channel, out_shape)
        for f in [in_channel] + list(out_shape):
            in_feature *= f
        for i, out_feature in enumerate(features):
            # print("%s_linear_%d" % (event, i), in_feature, out_feature)
            layers["%s_linear_%d" % (event, i)] = nn.Linear(in_feature,
                                                            out_feature)
            # layers["%s_linear_batch_norm_%d" %
            #        (event, i)] = nn.BatchNorm1d(out_feature)
            layers["%s_linear_relu_%d" % (event, i)] = nn.ReLU()
            layers["%s_linear_dropout_%d" % (event, i)] = nn.Dropout()
            in_feature = out_feature
        layers[event] = nn.Linear(out_feature, 1)
        # layers[event] = nn.Softmax(dim=-1)  # to probability # Unit1の場合、すべて1になる
        # layers[event] = nn.Sigmoid()  # to 0-1.0 values

        return nn.Sequential(layers), event
