using NeuronalNetwork.Layers;
using NeuronalNetwork.Libraries;
using NeuronalNetwork.Neurons;
using System;
using System.Collections.Generic;

namespace NeuronalNetwork
{
    public class NeuronalNetwork
    {
        /// <summary>
        /// Doesn't include input layer
        /// </summary>
        public List<Layer> layers;

        public readonly ActivationFunctions activationFunction;
        public readonly CostFunctions costFunction;
        public readonly int inputLenght;

        public NeuronalNetwork(int inputLenght, ActivationFunctions activationFunction, CostFunctions costFunction = CostFunctions.SquaredMean)
        {
            this.activationFunction = activationFunction;
            layers = new List<Layer>();
            this.inputLenght = inputLenght;
            this.costFunction = costFunction;
        }

        public virtual double[] ExecuteNetwork(double[] input, out List<double[]> neuronActivations)
        {
            if (inputLenght != input.Length)
                throw new Exception("Input lenght is different than the defined lenght");

            neuronActivations = new List<double[]>();
            //input layer
            double[] output = input;

            //other layers
            for (int layerIndex = 0; layerIndex < layers.Count; layerIndex++)
                neuronActivations.Add(output = layers[layerIndex].ExecuteLayer(output, activationFunction));

            return output;
        }

        public double[] ExecuteNetwork(double[] input) => ExecuteNetwork(input, out _);

        public void AddLayer(int lenght, LayerValues weights = null)
        {
            int previousNeurons = layers.Count == 0 ?
                inputLenght : layers[layers.Count - 1].length;

            layers.Add(new Layer(previousNeurons, lenght, weights));
        }

        public enum ActivationFunctions
        {
            Relu,
            Sigmoid,
            Tanh,
        }

        public enum CostFunctions
        {
            SquaredMean,
            BinaryCrossEntropy,
        }

        #region Training

        /*public void BackPropagationBatch(List<double[]> inputs, List<double[]> expectedOutputs, out double averageCost)
        {
            averageCost
        }*/

        public virtual List<LayerValues> GetGradients(double[] input, double[] expectedOutput, ActivationFunctions activation = ActivationFunctions.Relu, CostFunctions cost = CostFunctions.SquaredMean)
        {
            List<LayerValues> networkGradients = new List<LayerValues>();
            double[] networkOutput = ExecuteNetwork(input);

            double[] costGradients = new double[layers[layers.Count - 1].length];
            double[] previousLayerActivationsGradients = new double[layers[layers.Count - 2].length];
            double[] biasGradients = new double[layers.Count];

            //layers
            for (int layerIndex = layers.Count - 1; layerIndex <= 1; layerIndex--)
            {
                int layerLenght = layers[layerIndex].length, previousLayerLenght = layers[layerIndex - 1].length;

                double[] activationFunctionGradients = new double[layerLenght];
                double[] weightGradients = new double[layerLenght * previousLayerLenght];

                //neurons
                for (int neuronIndex = 0; neuronIndex < layerLenght; neuronIndex++)
                {
                    if (layerIndex == layers.Count - 1)
                        switch (cost)
                        {
                            case CostFunctions.SquaredMean:
                                costGradients[neuronIndex] = Derivatives.SquaredMeanErrorDerivative(networkOutput[neuronIndex], expectedOutput[neuronIndex]);
                                break;

                            case CostFunctions.BinaryCrossEntropy:
                                
                                break;

                            default:
                                break;
                        }

                    double startingGradient = layerIndex == layers.Count - 1 ? costGradients[neuronIndex] : previousLayerActivationsGradients[neuronIndex];

                    //weights
                    for (int weightIndex = 0; weightIndex < previousLayerLenght; weightIndex++)
                    {
                        double currentActivation = layers[layerIndex].neurons[neuronIndex].lastActivation;

                        //
                        switch (activation)
                        {
                            case ActivationFunctions.Relu:
                                activationFunctionGradients[neuronIndex] = Derivatives.ReluDerivative(currentActivation);
                                break;

                            case ActivationFunctions.Sigmoid:
                                activationFunctionGradients[neuronIndex] = Derivatives.SigmoidActivationDerivative(currentActivation);
                                break;

                            case ActivationFunctions.Tanh:
                                activationFunctionGradients[neuronIndex] = Derivatives.TanhDerivative(currentActivation);
                                break;

                            default:
                                throw new NotImplementedException();
                        }
                        double linearFunctionGradient = layers[layerIndex - 1].neurons[weightIndex].lastActivation;

                        //
                        weightGradients[weightIndex] = Gradients.WeightGradient(
                            startingGradient,
                            activationFunctionGradients[neuronIndex],
                            linearFunctionGradient
                            );

                        //
                        previousLayerActivationsGradients[weightIndex] += Gradients.ConnectedNeuronGradient(
                            startingGradient,
                            activationFunctionGradients[neuronIndex],
                            layers[layerIndex].neurons[neuronIndex].weights[weightIndex]
                            );

                        networkGradients[layerIndex].neurons[neuronIndex].weights[weightIndex] = weightGradients[weightIndex];
                    }
                    //
                    biasGradients[neuronIndex] += Gradients.BiasGradient(startingGradient, activationFunctionGradients[neuronIndex]);
                    networkGradients[layerIndex].bias = biasGradients[neuronIndex];
                }
            }
            return networkGradients;
        }

        public void ApplyGradients(List<LayerValues> gradients, double learningRate = 1.5)
        {
            for (int layerIndex = 0; layerIndex < layers.Count; layerIndex++)
            {
                for (int neuronIndex = 0; neuronIndex < layers[layerIndex].length; neuronIndex++)
                {
                    Neuron currentNeuron = layers[layerIndex].neurons[neuronIndex];
                    for (int weightIndex = 0; weightIndex < currentNeuron.weights.Length; weightIndex++)
                    {
                        layers[layerIndex].neurons[neuronIndex].weights[weightIndex]
                            -= gradients[layerIndex].neurons[neuronIndex].weights[weightIndex] * learningRate;
                    }
                }
            }
        }

        public static class Derivatives
        {
            public static double SquaredMeanErrorDerivative(double neuronOutput, double expectedOutput) => 2 * (neuronOutput - expectedOutput);

            //public static double BinaryCrossEntropyDerivative(double neuronOutput, double expectedOutput) =>  /

            public static double SigmoidActivationDerivative(double neuronActivation) => Neuron.SigmoidActivation(neuronActivation) * (1 - SigmoidActivationDerivative(neuronActivation));

            /// <param name="connectedNeuronActivation">Activation Connected to the weight that is being computed</param>
            public static double LinearFunctionDerivative(double connectedNeuronActivation) => connectedNeuronActivation;

            public static int ReluDerivative(double neuronActivation) => neuronActivation > 0 ? 1 : 0;

            public static double TanhDerivative(double neuronActivation) => 1 - Math.Pow(Neuron.TanhActivation(neuronActivation), 2);
        }

        public static class Gradients
        {
            public static double WeightGradient(double costGradient, double activationFunctionGradient, double previousNeuronActivation)
                => costGradient * activationFunctionGradient * previousNeuronActivation;

            public static double ConnectedNeuronGradient(double costGradient, double activationFunctionGradient, double connectedWeight)
                => costGradient * activationFunctionGradient * connectedWeight;

            public static double BiasGradient(double costGradient, double activationFunctionGradient)
                => costGradient * activationFunctionGradient;
        }

        #endregion Training
    }
}