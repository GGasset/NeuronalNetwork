using NeuronalNetwork.Libraries;
using NeuronalNetwork.Neurons;
using System.Collections.Generic;

namespace NeuronalNetwork.Layers
{
    public class Layer
    {
        public int length;
        public double bias;
        public List<Neuron> neurons;

        public Layer(int previousLayerLenght, int length, LayerValues weights = null)
        {
            neurons = new List<Neuron>();

            if (weights != null)
                for (int i = 0; i < weights.neurons.Count; i++)
                    neurons.Add(new Neuron(previousLayerLenght, weights.GetWeights(i)));
            else
                for (int i = 0; i < weights.neurons.Count; i++)
                    neurons.Add(new Neuron(previousLayerLenght));

            bias = weights.bias;
            this.length = length;
        }

        public virtual double[] ExecuteLayer(double[] previousLayerActivations, NeuronalNetwork.ActivationFunctions activation)
        {
            double[] currentActivations = new double[neurons.Count];
            for (int i = 0; i < neurons.Count; i++)
            {
                currentActivations[i] = neurons[i].ExecuteNeuron(previousLayerActivations, bias, activation);
            }
            return currentActivations;
        }
    }
}