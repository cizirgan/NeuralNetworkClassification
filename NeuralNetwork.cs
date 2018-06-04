using System;

namespace Haberman
{
    class NeuralNetwork
    {
        private int numInput;
        private int numHidden;
        private int numOutput;

        private double[] inputs;
        private double[][] ihWeights; // input-to-hidden
        private double[] ihSums;
        private double[] ihBiases;
        private double[] ihOutputs;
        private double[][] hoWeights;  // hidden-to-output
        private double[] hoSums;
        private double[] hoBiases;
        private double[] outputs;

        static Random rnd = null;

        public NeuralNetwork(int numInput, int numHidden, int numOutput)
        {
            this.numInput = numInput;
            this.numHidden = numHidden;
            this.numOutput = numOutput;

            inputs = new double[numInput];
            ihWeights = Helpers.MakeMatrix(numInput, numHidden);
            ihSums = new double[numHidden];
            ihBiases = new double[numHidden];
            ihOutputs = new double[numHidden];
            hoWeights = Helpers.MakeMatrix(numHidden, numOutput);
            hoSums = new double[numOutput];
            hoBiases = new double[numOutput];
            outputs = new double[numOutput];

            rnd = new Random(0);
        }

        public void SetWeights(double[] weights)
        {
            int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
            if (weights.Length != numWeights)
                throw new Exception("The weights array length: " + weights.Length + " does not match the total number of weights and biases: " + numWeights);

            int k = 0; // points into weights param

            for (int i = 0; i < numInput; ++i)
                for (int j = 0; j < numHidden; ++j)
                    ihWeights[i][j] = weights[k++];

            for (int i = 0; i < numHidden; ++i)
                ihBiases[i] = weights[k++];

            for (int i = 0; i < numHidden; ++i)
                for (int j = 0; j < numOutput; ++j)
                    hoWeights[i][j] = weights[k++];

            for (int i = 0; i < numOutput; ++i)
                hoBiases[i] = weights[k++];
        }

        public double[] ComputeOutputs(double[] currInputs)
        {
            if (inputs.Length != numInput)
                throw new Exception("Inputs array length " + inputs.Length + " does not match NN numInput value " + numInput);

            for (int i = 0; i < numHidden; ++i)
                this.ihSums[i] = 0.0;
            //for (int i = 0; i < numHidden; ++i)
            //  this.ihOutputs[i] = 0.0;
            for (int i = 0; i < numOutput; ++i)
                this.hoSums[i] = 0.0;
            //for (int i = 0; i < numOutput; ++i)
            //  this.outputs[i] = 0.0;


            for (int i = 0; i < currInputs.Length; ++i) // copy
                this.inputs[i] = currInputs[i];

            //Console.WriteLine("Inputs:");
            //ShowVector(this.inputs);

            //Console.WriteLine("input-to-hidden weights:");
            //ShowMatrix(this.ihWeights);

            for (int j = 0; j < numHidden; ++j)  // compute input-to-hidden sums
                for (int i = 0; i < numInput; ++i)
                    ihSums[j] += this.inputs[i] * ihWeights[i][j];

            //Console.WriteLine("input-to-hidden sums:");
            //ShowVector(this.ihSums);

            //Console.WriteLine("input-to-hidden biases:");
            //ShowVector(ihBiases);

            for (int i = 0; i < numHidden; ++i)  // add biases to input-to-hidden sums
                ihSums[i] += ihBiases[i];

            //Console.WriteLine("input-to-hidden sums after adding biases:");
            //ShowVector(this.ihSums);

            for (int i = 0; i < numHidden; ++i)   // determine input-to-hidden output
                                                  //ihOutputs[i] = StepFunction(ihSums[i]); // step function
                ihOutputs[i] = SigmoidFunction(ihSums[i]);
            //ihOutputs[i] = TanhFunction(ihSums[i]);

            //Console.WriteLine("input-to-hidden outputs after sigmoid:");
            //ShowVector(this.ihOutputs);

            //Console.WriteLine("hidden-to-output weights:");
            //ShowMatrix(hoWeights);


            for (int j = 0; j < numOutput; ++j)   // compute hidden-to-output sums
                for (int i = 0; i < numHidden; ++i)
                    hoSums[j] += ihOutputs[i] * hoWeights[i][j];

            //Console.WriteLine("hidden-to-output sums:");
            //ShowVector(hoSums);

            //Console.WriteLine("hidden-to-output biases:");
            //ShowVector(this.hoBiases);

            for (int i = 0; i < numOutput; ++i)  // add biases to input-to-hidden sums
                hoSums[i] += hoBiases[i];

            //Console.WriteLine("hidden-to-output sums after adding biases:");
            //ShowVector(this.hoSums);

            //for (int i = 0; i < numOutput; ++i)   // determine hidden-to-output result
            //  this.outputs[i] = SigmoidFunction(hoSums[i]);  // step function

            //double[] result = new double[numOutput];
            //this.outputs.CopyTo(result, 0);
            //return result;

            double[] result = Softmax(hoSums);

            result.CopyTo(this.outputs, 0);

            //Console.WriteLine("outputs after softmaxing:");
            //ShowVector(result);

            //Console.ReadLine();

            //double[] result = Hardmax(hoSums);
            return result;
        } // ComputeOutputs

        //private static double StepFunction(double x)
        //{
        //  if (x > 0.0) return 1.0;
        //  else return 0.0;
        //}

        private static double SigmoidFunction(double x)
        {
            if (x < -45.0) return 0.0;
            else if (x > 45.0) return 1.0;
            else return 1.0 / (1.0 + Math.Exp(-x));
        }

        private static double[] Softmax(double[] hoSums)
        {
            // determine max
            double max = hoSums[0];
            for (int i = 0; i < hoSums.Length; ++i)
                if (hoSums[i] > max) max = hoSums[i];

            // determine scaling factor (sum of exp(eachval - max)
            double scale = 0.0;
            for (int i = 0; i < hoSums.Length; ++i)
                scale += Math.Exp(hoSums[i] - max);

            double[] result = new double[hoSums.Length];
            for (int i = 0; i < hoSums.Length; ++i)
                result[i] = Math.Exp(hoSums[i] - max) / scale;

            return result;
        }

        public double[] Train(double[][] trainMatrix) // seek and return the best weights
        {
            int numWeights = (this.numInput * this.numHidden) + (this.numHidden * this.numOutput) + this.numHidden + this.numOutput;
            //double[] currWeights = new double[numWeights];

            // use PSO to seek best weights
            int numberParticles = 10;
            int numberIterations = 500;
            int iteration = 0;
            int Dim = numWeights; // number of values to solve for
            double minX = -5.0; // for each weight
            double maxX = 5.0;

            Particle[] swarm = new Particle[numberParticles];
            double[] bestGlobalPosition = new double[Dim]; // best solution found by any particle in the swarm. implicit initialization to all 0.0
            double bestGlobalFitness = double.MaxValue; // smaller values better

            double minV = -0.1 * maxX;  // velocities
            double maxV = 0.1 * maxX;

            for (int i = 0; i < swarm.Length; ++i) // initialize each Particle in the swarm with random positions and velocities
            {
                double[] randomPosition = new double[Dim];
                for (int j = 0; j < randomPosition.Length; ++j)
                {
                    double lo = minX;
                    double hi = maxX;
                    randomPosition[j] = (hi - lo) * rnd.NextDouble() + lo;
                }

                double fitness = CrossEntropy(trainMatrix, randomPosition); // smaller values better
                double[] randomVelocity = new double[Dim];

                for (int j = 0; j < randomVelocity.Length; ++j)
                {
                    double lo = -1.0 * Math.Abs(maxX - minX);
                    double hi = Math.Abs(maxX - minX);
                    randomVelocity[j] = (hi - lo) * rnd.NextDouble() + lo;
                }
                swarm[i] = new Particle(randomPosition, fitness, randomVelocity, randomPosition, fitness);

                // does current Particle have global best position/solution?
                if (swarm[i].fitness < bestGlobalFitness)
                {
                    bestGlobalFitness = swarm[i].fitness;
                    swarm[i].position.CopyTo(bestGlobalPosition, 0);
                }
            } // initialization

            double w = 0.729; // inertia weight.
            double c1 = 1.49445; // cognitive/local weight
            double c2 = 1.49445; // social/global weight
            double r1, r2; // cognitive and social randomizations

            Console.WriteLine("Entering main PSO weight estimation processing loop");
            while (iteration < numberIterations)
            {
                ++iteration;
                double[] newVelocity = new double[Dim];
                double[] newPosition = new double[Dim];
                double newFitness;

                for (int i = 0; i < swarm.Length; ++i) // each Particle
                {
                    Particle currP = swarm[i];

                    for (int j = 0; j < currP.velocity.Length; ++j) // each x value of the velocity
                    {
                        r1 = rnd.NextDouble();
                        r2 = rnd.NextDouble();

                        newVelocity[j] = (w * currP.velocity[j]) +
                          (c1 * r1 * (currP.bestPosition[j] - currP.position[j])) +
                          (c2 * r2 * (bestGlobalPosition[j] - currP.position[j])); // new velocity depends on old velocity, best position of parrticle, and best position of any particle

                        if (newVelocity[j] < minV)
                            newVelocity[j] = minV;
                        else if (newVelocity[j] > maxV)
                            newVelocity[j] = maxV;     // crude way to keep velocity in range
                    }

                    newVelocity.CopyTo(currP.velocity, 0);

                    for (int j = 0; j < currP.position.Length; ++j)
                    {
                        newPosition[j] = currP.position[j] + newVelocity[j];  // compute new position
                        if (newPosition[j] < minX)
                            newPosition[j] = minX;
                        else if (newPosition[j] > maxX)
                            newPosition[j] = maxX;
                    }

                    newPosition.CopyTo(currP.position, 0);

                    newFitness = CrossEntropy(trainMatrix, newPosition);  // compute error of the new position
                    currP.fitness = newFitness;

                    if (newFitness < currP.bestFitness) // new particle best?
                    {
                        newPosition.CopyTo(currP.bestPosition, 0);
                        currP.bestFitness = newFitness;
                    }

                    if (newFitness < bestGlobalFitness) // new global best?
                    {
                        newPosition.CopyTo(bestGlobalPosition, 0);
                        bestGlobalFitness = newFitness;
                    }

                } // each Particle

                //Console.WriteLine(swarm[0].ToString());
                //Console.ReadLine();

            } // while

            Console.WriteLine("Processing complete");
            Console.Write("Final best (smallest) cross entropy error = ");
            Console.WriteLine(bestGlobalFitness.ToString("F4"));

            return bestGlobalPosition;

        } // Train

        private double CrossEntropy(double[][] trainData, double[] weights) // (sum) Cross Entropy
        {
            // how good (cross entropy) are weights? CrossEntropy is error so smaller values are better
            this.SetWeights(weights); // load the weights and biases to examine

            double sce = 0.0; // sum of cross entropy

            for (int i = 0; i < trainData.Length; ++i) // walk thru each training case. looks like (6.9 3.2 5.7 2.3) (0 0 1)  where the parens are not really there
            {
                double[] currInputs = new double[3]; 
                currInputs[0] = trainData[i][0]; 
                currInputs[1] = trainData[i][1]; 
                currInputs[2] = trainData[i][2]; 
   
                double[] currExpected = new double[2]; 
                currExpected[0] = trainData[i][3]; 
                currExpected[1] = trainData[i][4]; 


                double[] currOutputs = this.ComputeOutputs(currInputs); // run the jnputs through the neural network

                // compute ln of each nn output (and the sum)
                double currSum = 0.0;
                for (int j = 0; j < currOutputs.Length; ++j)
                {
                    if (currExpected[j] != 0.0)
                        currSum += currExpected[j] * Math.Log(currOutputs[j]);
                }
                sce += currSum; // accumulate
            }
            return -sce;
        } // CrossEntropy

        public double Test(double[][] testMatrix) // returns the accuracy (percent correct predictions)
        {
            // assumes that weights have been set using SetWeights
            int numCorrect = 0;
            int numWrong = 0;

            for (int i = 0; i < testMatrix.Length; ++i) // walk thru each test case. looks like (6.9 3.2 5.7 2.3) (0 0 1)  where the parens are not really there
            {

                double[] currInputs = new double[3];
                currInputs[0] = testMatrix[i][0];
                currInputs[1] = testMatrix[i][1];
                currInputs[2] = testMatrix[i][2];
                double[] currOutputs = new double[2];
                currOutputs[0] = testMatrix[i][3];
                currOutputs[1] = testMatrix[i][4];

                double[] currPredicted = this.ComputeOutputs(currInputs); // outputs are in softmax form -- each between 0.0, 1.0 representing a prob and summing to 1.0

                //ShowVector(currInputs);
                //ShowVector(currOutputs);
                //ShowVector(currPredicted);

                // use winner-takes all -- highest prob of the prediction
                int indexOfLargest = Helpers.IndexOfLargest(currPredicted);

                if (i <= 3) // just a few for demo purposes
                {
                    Console.WriteLine("-----------------------------------");
                    Console.Write("Input:     ");
                    Helpers.ShowVector(currInputs, 2, true);
                    Console.Write("Output:    ");
                    Helpers.ShowVector(currOutputs, 1, false);
                    if (currOutputs[0] == 1.0) Console.WriteLine("survived 5 years or longer");
                    else if (currOutputs[1] == 1.0) Console.WriteLine("died within 5 year");

                    Console.Write("Predicted: ");
                    Helpers.ShowVector(currPredicted, 1, false);
                    if (indexOfLargest == 0) Console.WriteLine("survived 5 years or longer");
                    else if (indexOfLargest == 1) Console.WriteLine("died within 5 year");


                    if (currOutputs[indexOfLargest] == 1)
                        Console.WriteLine("correct");
                    else
                        Console.WriteLine("wrong");
                    Console.WriteLine("-----------------------------------");
                }

                if (currOutputs[indexOfLargest] == 1)
                    ++numCorrect;
                else
                    ++numWrong;

                //Console.ReadLine();
            }
            Console.WriteLine(". . .");

            double percentCorrect = (numCorrect * 1.0) / (numCorrect + numWrong);
            Console.WriteLine("\nCorrect = " + numCorrect);
            Console.WriteLine("Wrong = " + numWrong);

            return percentCorrect;
        } // Test

    }
}