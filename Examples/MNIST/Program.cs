using System;
using System.Threading.Tasks;
using NumSharp;
using Tensorflow;
using Tensorflow.Hub;
using static Tensorflow.Binding;

namespace MNIST
{
    class Program
    {
        static async Task Main(string[] args)
        {
            NDArray x_train, y_train;
            NDArray x_valid, y_valid;
            NDArray x_test, y_test;

            var mnist = await MnistModelLoader.LoadAsync(".resources/fashion_mnist", oneHot: true);
            x_train = mnist.Train.Data;
            y_train = mnist.Train.Labels;
            x_valid = mnist.Validation.Data;
            y_valid = mnist.Validation.Labels;
            x_test = mnist.Test.Data;
            y_test = mnist.Test.Labels;

            var model = keras.Sequential();
            model.add(keras.layers.Dense(28*28));
            model.add(keras.layers.Dense(128, activation: "relu"));
            model.add(keras.layers.Dense(10, activation: "softmax"));
        }
    }
}
