using NeuronalNetwork.Libraries;
using NeuronalNetwork.Neurons;
using System.Collections.Generic;

namespace NeuronalNetwork.Layers
{
    internal class RecurrentLayer : Layer
    {
        new public List<RecurrentNeuron> neurons = new List<RecurrentNeuron>();

        public RecurrentLayer(int previousLayerLenght, int length, LayerValues layerWeights = null)
                        : base(previousLayerLenght, length, layerWeights)
        {
            if (layerWeights != null)
                for (int i = 0; i < length; i++)
                    neurons.Add(new RecurrentNeuron(previousLayerLenght, layerWeights.GetWeights(i), layerWeights.GetRecurrent(i)));
            else
                for (int i = 0; i < length; i++)
                    neurons.Add(new RecurrentNeuron(previousLayerLenght, null));
        }

        public override double[] ExecuteLayer(double[] previousLayerActivations, NeuronalNetwork.ActivationFunctions activation)
        {
            double[] activations = new double[length];
            for (int i = 0; i < neurons.Count; i++)
                activations[i] = neurons[i].ExecuteNeuron(activations, bias, activation);

            return activations;
        }
    }
}