using System;
using NeuronalNetwork;
using System.Collections.Generic;

namespace Gmoty
{
    public class Gmoty
    {



        public Gmoty()
        {

        }
    }

    public class NetworkHolder
    {
        internal List<TemporalNetwork> networks;
        internal List<NetworkConnections> connections;

        public NetworkHolder(NetworkHolder networks)
        {
            this.networks = networks.networks;
            connections = networks.connections;
        }

        /*public override string ToString()
        {

        }

        public static NetworkHolder FromString(string value)
        {

        }*/
    }

    public class NetworkConnections
    {
        internal int fromNetworkIndex, toNetworkIndex;
        internal Range fromOutputRange, toInputRange;
    }

    public class Range
    {
        public int start, end;

        public Range(int start, int end)
        {
            this.start = start;
            this.end = end;
        }
    }
}
