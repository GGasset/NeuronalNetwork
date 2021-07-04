using System;

namespace NeuronalNetwork.Neurons
{
    class RecurrentNeuron : Neuron
    {
        public double recurrentWeight;

        public RecurrentNeuron(int previousLayerLenght, double[] weights = null, double weightToItself = double.MinValue)
                               : base(previousLayerLenght, weights)
        {
            if (weightToItself != double.MinValue)
                recurrentWeight = weightToItself;
            else
                recurrentWeight = GetRandomWeight();
        }

        public override double ExecuteNeuron(double[] input, double bias, NeuronalNetwork.ActivationFunctions activation)
        {
            if (input.Length != weights.Length)
                throw new IndexOutOfRangeException();

            double output = lastActivation * recurrentWeight;
            for (int i = 0; i < input.Length; i++)
                output += input[i] * weights[i];

            output = activation switch
            {
                NeuronalNetwork.ActivationFunctions.Relu => ReluActivation(output),
                NeuronalNetwork.ActivationFunctions.Sigmoid => SigmoidActivation(output),
                NeuronalNetwork.ActivationFunctions.Tanh => TanhActivation(output),
                _ => throw new NotImplementedException(),
            };

            return output;
        }
    }
}
