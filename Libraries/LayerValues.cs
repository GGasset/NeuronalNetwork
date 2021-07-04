using System.Collections.Generic;

namespace NeuronalNetwork.Libraries
{
    public class LayerValues
    {
        public List<NeuronWeights> neurons;
        public double bias;

        public LayerValues(double bias = 1)
        {
            neurons = new List<NeuronWeights>();
            this.bias = bias;
        }

        public double[] GetWeights(int neuronIndex) => neurons[neuronIndex].weights.ToArray();
        public double[] GetAllWeights()
        {
            List<double> output = new List<double>();
            for (int i = 0; i < neurons.Count; i++)
                output.AddRange(GetWeights(i));
            return output.ToArray();
        }
        public double GetRecurrent(int index) => neurons[index].recurrentWeight;

        public class NeuronWeights
        {
            public List<double> weights;
            public double recurrentWeight = double.MinValue;

            public NeuronWeights()
            {
                weights = new List<double>();
            }

            public NeuronWeights(double[] weights)
            {
                this.weights = new List<double>();
                this.weights.AddRange(weights);
            }

            public NeuronWeights(double[] weights, double recurrentWeight)
            {
                this.weights = new List<double>();
                this.weights.AddRange(weights);
                this.recurrentWeight = recurrentWeight;
            }
        }
    }
}
