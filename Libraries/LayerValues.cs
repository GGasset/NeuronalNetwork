using System;
using System.Collections.Generic;

namespace NeuronalNetwork.Libraries
{
    public class NetworkValues
    {
        public List<LayerValues> values;

        public NetworkValues()
        {
            values = new List<LayerValues>();
        }

        public NetworkValues(List<double[]>[] values)
        {
            List<LayerValues> output = new List<LayerValues>();
            for (int i = 0; i < values.Length; i++)
                output.Add(new LayerValues(values[i]));

            this.values = output;
        }

        public LayerValues this[int index] => values[index];
    }

    public class LayerValues
    {
        public List<NeuronWeights> neurons;
        public double bias;

        public LayerValues(double bias = 1)
        {
            neurons = new List<NeuronWeights>();
            this.bias = bias;
        }

        public LayerValues(List<double[]> values, double bias = 1)
        {
            for (int i = 0; i < values.Count; i++)
                neurons.Add(new NeuronWeights(values[i]));
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
            public double recurrentWeight = 0;
            public int lenght => weights.Count;

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

            public static NeuronWeights operator +(NeuronWeights a, NeuronWeights b)
            {
                double recurrentWeight = a.recurrentWeight + b.recurrentWeight;
                return new NeuronWeights(AddArrays(a.weights.ToArray(), b.weights.ToArray()), recurrentWeight);
            }

            public static double[] AddArrays(double[] a, double[] b)
            {
                int maxLenght = Math.Max(a.Length, b.Length);
                int minLenght = Math.Min(a.Length, b.Length);
                double[] output = new double[maxLenght];
                
                for (int i = 0; i < minLenght; i++)
                    output[i] = a[i] + b[i];

                if (a.Length == maxLenght)
                    for (int i = 0; i < maxLenght - minLenght; i++)
                        output[i + minLenght] = a[i];
                else
                    for (int i = 0; i < maxLenght - minLenght; i++)
                        output[i + minLenght] = b[i];

                return output;
            }

        }
    }
}
