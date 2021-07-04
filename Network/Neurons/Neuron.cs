using System;

namespace NeuronalNetwork.Neurons
{
    public class Neuron
    {
        internal double[] weights;
        internal double lastActivation = 0;

        public Neuron(int previousLayerLenght, double[] weights = null)
        {
            if (weights != null)
                this.weights = weights;
            else
                for (int i = 0; i < previousLayerLenght; i++)
                    weights[i] = GetRandomWeight();
        }

        public virtual double ExecuteNeuron(double[] input, double bias, NeuronalNetwork.ActivationFunctions activation)
        {
            if (input.Length != weights.Length)
                throw new IndexOutOfRangeException();

            double output = bias;
            for (int i = 0; i < input.Length; i++)
                output += input[i] * weights[i];

            output = activation switch
            {
                NeuronalNetwork.ActivationFunctions.Relu => ReluActivation(output),
                NeuronalNetwork.ActivationFunctions.Sigmoid => SigmoidActivation(output),
                NeuronalNetwork.ActivationFunctions.Tanh => TanhActivation(output),
                _ => throw new NotImplementedException(),
            };
            return lastActivation = output;
        }

        public static double ReluActivation(double input) => Math.Max(0, input);

        public static double SigmoidActivation(double input) => 1.0 / (1 + Math.Exp(-input));

        public static double TanhActivation(double input) => (Math.Exp(input) - Math.Exp(-input)) / (Math.Exp(input) + Math.Exp(-input));

        public double GetRandomWeight()
        {
            Random r = new Random();
            return r.Next(-3, 4) + r.NextDouble() * r.Next(-1, 2);
        }
    }
}