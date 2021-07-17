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

        public NetworkValues GetGradients(double[] input, double[] expectedOutput, out double cost, ActivationFunctions activation = ActivationFunctions.Relu, CostFunctions costFunction = CostFunctions.SquaredMean, double dropoutRate = .4)
        {
            NetworkValues inversedNetworkGradients = new NetworkValues();
            double[] networkOutput = ExecuteNetwork(input);

            cost = Cost.GetCostOf(networkOutput, expectedOutput, costFunction);
            double[] previousLayerActivationsGradients = new double[layers[layers.Count-1].length];

            //layers
            for (int layerIndex = layers.Count - 1; layerIndex <= 1; layerIndex--)
            {
                if (layerIndex == layers.Count - 1)
                    for (int neuronIndex = 0; neuronIndex < layers[layerIndex].length; neuronIndex++)
                    //cost gradients or initial layer 
                        previousLayerActivationsGradients[neuronIndex] = Derivatives.DerivativeOf(networkOutput[neuronIndex], expectedOutput[neuronIndex], costFunction);

                inversedNetworkGradients.values.Add(CalculateLayerGradients(previousLayerActivationsGradients, out previousLayerActivationsGradients, layerIndex, activation, dropoutRate));
            }

            NetworkValues Gradients = new NetworkValues();
            for (int i = inversedNetworkGradients.values.Count - 1; i >= 0; i--)
                Gradients.values.Add(inversedNetworkGradients.values[i]);

            return Gradients;
        }
        
        public LayerValues CalculateLayerGradients(double[] neuronActivationGradients, out double[] previousLayerActivationGradients, int layerIndex, ActivationFunctions activation = ActivationFunctions.Sigmoid, double dropoutRate = .4)
        {
            LayerValues output = new LayerValues();
            int layerLenght = layers[layerIndex].length, previousLayerLenght = layers[layerIndex - 1].length;

            double[] activationFunctionGradients = new double[layerLenght];
            double[] weightGradients = new double[layerLenght * previousLayerLenght];
            double[] biasGradients = new double[layers.Count];
            previousLayerActivationGradients = new double[previousLayerLenght];

            List<bool> previousLayerDropout = new List<bool>();
            for (int i = 0; i < previousLayerLenght; i++)// Set dropout
                if (new Random().NextDouble() < dropoutRate)
                    previousLayerDropout.Add(true);
                else
                    previousLayerDropout.Add(false);


            //neurons
            for (int neuronIndex = 0; neuronIndex < layerLenght; neuronIndex++)
            {

                double currentNeuronGradient = neuronActivationGradients[neuronIndex];

                //weights
                for (int weightIndex = 0; weightIndex < previousLayerLenght; weightIndex++)
                {
                    while (previousLayerDropout[weightIndex])
                    //Skip dropped out Neurons
                        weightIndex++;


                    double currentActivation = layers[layerIndex].neurons[neuronIndex].lastActivation;

                    //
                    activationFunctionGradients[neuronIndex] = Derivatives.DerivativeOf(currentActivation, activation);


                    double linearFunctionGradient = layers[layerIndex - 1].neurons[weightIndex].lastActivation;
                    //
                    weightGradients[weightIndex] = Gradients.WeightGradient(
                        currentNeuronGradient,
                        activationFunctionGradients[neuronIndex],
                        linearFunctionGradient
                        );

                    //
                        previousLayerActivationGradients[weightIndex] += Gradients.ConnectedNeuronGradient(
                            currentNeuronGradient,
                            activationFunctionGradients[neuronIndex],
                            layers[layerIndex].neurons[neuronIndex].weights[weightIndex]
                            );

                    output.neurons[neuronIndex].weights[weightIndex] = weightGradients[weightIndex];
                }
                //
                biasGradients[neuronIndex] += Gradients.BiasGradient(currentNeuronGradient, activationFunctionGradients[neuronIndex]);
                output.bias = biasGradients[neuronIndex];
            }

            return output;
        }

        internal void ApplyGradients(List<LayerValues> gradients, double learningRate = 1.5)
        {
            for (int layerIndex = 0; layerIndex < layers.Count; layerIndex++)
                for (int neuronIndex = 0; neuronIndex < layers[layerIndex].length; neuronIndex++)
                {
                    Neuron currentNeuron = layers[layerIndex].neurons[neuronIndex];
                    for (int weightIndex = 0; weightIndex < currentNeuron.weights.Length; weightIndex++)
                        layers[layerIndex].neurons[neuronIndex].weights[weightIndex]
                            -= gradients[layerIndex].neurons[neuronIndex].weights[weightIndex] * learningRate;
                }
        }

        public static class Derivatives
        {
            public static double DerivativeOf(double neuronActivation, ActivationFunctions activation)
            {
                 return activation switch
                 {
                     ActivationFunctions.Relu => ReluDerivative(neuronActivation),
                     ActivationFunctions.Sigmoid => SigmoidActivationDerivative(neuronActivation),
                     ActivationFunctions.Tanh => TanhDerivative(neuronActivation),
                     _ => throw new NotImplementedException(),
                 };
            }

            public static double DerivativeOf(double neuronActivation, double expected, CostFunctions costFunction)
            {
                return costFunction switch
                {
                    CostFunctions.BinaryCrossEntropy => throw new NotImplementedException(),
                    CostFunctions.SquaredMean => SquaredMeanErrorDerivative(neuronActivation, expected),
                    _=> throw new NotImplementedException(), 
                };
            }

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

        public static class Cost
        {
            public static double GetCostOf(double[] output, double[] expected, CostFunctions costFunction)
            {
                return costFunction switch
                {
                    CostFunctions.BinaryCrossEntropy => BinaryCrossEntropyEmpiricalLoss(output, expected),
                    CostFunctions.SquaredMean => SquaredMeanEmpiricalLoss(output, expected),
                    _ => throw new NotImplementedException(),
                };
            }

            public static double SquaredMeanEmpiricalLoss(double[] output, double[] expected)
            {
                if (output.Length != expected.Length)
                    throw new Exception();
                double sum = 0;
                for (int i = 0; i < output.Length; i++)
                {
                    double currentCost = expected[i] - output[i];
                    sum += currentCost * currentCost;
                }
                sum /= output.Length;

                return sum;
            }

            /// <summary>
            /// Used to train the network for boolean outputs
            /// </summary>
            public static double BinaryCrossEntropyEmpiricalLoss(double[] output, double[] expected)
            {
                if (output.Length != expected.Length)
                    throw new Exception();
                double sum = 0;
                for (int i = 0; i < output.Length; i++)
                    sum += 1 - expected[i] * Math.Log(1 - output[i]) + expected[i] * Math.Log(expected[i]);
                sum /= output.Length;

                return sum;
            }
        }

        #endregion Training
    }
}