namespace Haberman
{
    public class Particle
    {
        public double[] position; // equivalent to x-Values and/or solution
        public double fitness;
        public double[] velocity;

        public double[] bestPosition; // best position found so far by this Particle
        public double bestFitness;

        public Particle(double[] position, double fitness, double[] velocity, double[] bestPosition, double bestFitness)
        {
            this.position = new double[position.Length];
            position.CopyTo(this.position, 0);
            this.fitness = fitness;
            this.velocity = new double[velocity.Length];
            velocity.CopyTo(this.velocity, 0);
            this.bestPosition = new double[bestPosition.Length];
            bestPosition.CopyTo(this.bestPosition, 0);
            this.bestFitness = bestFitness;
        }

        public override string ToString()
        {
            string s = "";
            s += "==========================\n";
            s += "Position: ";
            for (int i = 0; i < this.position.Length; ++i)
                s += this.position[i].ToString("F2") + " ";
            s += "\n";
            s += "Fitness = " + this.fitness.ToString("F4") + "\n";
            s += "Velocity: ";
            for (int i = 0; i < this.velocity.Length; ++i)
                s += this.velocity[i].ToString("F2") + " ";
            s += "\n";
            s += "Best Position: ";
            for (int i = 0; i < this.bestPosition.Length; ++i)
                s += this.bestPosition[i].ToString("F2") + " ";
            s += "\n";
            s += "Best Fitness = " + this.bestFitness.ToString("F4") + "\n";
            s += "==========================\n";
            return s;
        }

    }
}