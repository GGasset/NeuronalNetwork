using NeuronalNetwork.Layers;
using NeuronalNetwork.Libraries;
using System;
using System.Collections.Generic;

namespace NeuronalNetwork.Network
{
    internal class RecurrentNetwork : NeuronalNetwork
    {
        public List<LayerTypes> LayersType { get; private set; }
        public bool IsRecurrent { get; private set; }

        public RecurrentNetwork(int inputLenght, ActivationFunctions activation, CostFunctions costFunction = CostFunctions.SquaredMean)
            : base(inputLenght, activation, costFunction)
        {
            IsRecurrent = false;
            LayersType = new List<LayerTypes>();
        }

        public void AddLayer(int lenght, LayerTypes layerType, LayerValues weights = null)
        {
            int previousNeurons = layers.Count == 0 ?
                inputLenght : layers[layers.Count - 1].length;

            LayersType.Add(layerType);
            switch (layerType)
            {
                case LayerTypes.FeedForward:
                    layers.Add(new Layer(previousNeurons, lenght, weights));
                    break;

                case LayerTypes.Recurrent:
                    layers.Add(new RecurrentLayer(previousNeurons, lenght, weights));
                    IsRecurrent = true;
                    break;

                default:
                    throw new NotImplementedException();
            }
        }

        /// <summary>
        /// Execute the network multiple times, only makes sense in recurrent networks
        /// </summary>
        public double[] ExecuteNetwork(List<double[]> inputs, out NetworkValues neuronActivations)
        {
            List<double[]>[] unparsedNeuronActivations = new List<double[]>[inputs.Count];

            double[] output = ExecuteNetwork(inputs[0]);

            for (int i = 1; i < inputs.Count; i++)
                output = ExecuteNetwork(inputs[i], out unparsedNeuronActivations[i]);
            neuronActivations = new NetworkValues(unparsedNeuronActivations);

            if (IsRecurrent)
                ResetHiddenStates();

            return output;
        }

        private void ResetHiddenStates()
        {
            for (int layerIndex = 0; layerIndex < layers.Count; layerIndex++)
                if (LayersType[layerIndex] != LayerTypes.FeedForward)
                    for (int neuronIndex = 0; neuronIndex < layers[layerIndex].length; neuronIndex++)
                        layers[layerIndex].neurons[neuronIndex].lastActivation = 0;
        }

        public enum LayerTypes
        {
            FeedForward,
            Recurrent,
        }

        #region Training

        

        #endregion Training
    }
}