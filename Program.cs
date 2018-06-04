using System;

namespace Haberman
{
    class Program
    {
        static Random rnd = null;
        static void Main(string[] args)
        {   
            rnd = new Random(159);
            string dataFile = "colors.txt";
            Helpers.MakeData(dataFile, 100, rnd);


            Helpers.ShowTextFile(dataFile, 4);

            double[][] trainMatrix = null;
            double[][] testMatrix = null;

            //Console.WriteLine("\nGenerating train and test matrices using an 80%-20% split");
           // Operations.MakeTrainAndTest(dataFile, out trainMatrix, out testMatrix);

            string dataFileHaberman = "haberman.data";
            Operations.MakeTrainAndTest(dataFileHaberman, out trainMatrix, out testMatrix);

            Console.WriteLine("\nFirst few rows of training matrix are:");
            Helpers.ShowMatrix(trainMatrix, 50);

            Console.WriteLine("\nCreating 4-input 5-hidden 3-output neural network");
            NeuralNetwork nn = new NeuralNetwork(3, 4, 2);

            Console.WriteLine("Training to find best neural network weights using PSO with cross entropy error");
            double[] bestWeights = nn.Train(trainMatrix);
            Console.WriteLine("\nBest weights found:");
            Helpers.ShowVector(bestWeights, 2, true);

            Console.WriteLine("\nLoading best weights into neural network");
            nn.SetWeights(bestWeights);

            Console.WriteLine("\nAnalyzing the neural network accuracy on the test data\n");
            double accuracy = nn.Test(testMatrix);
            Console.WriteLine("Prediction accuracy = " + accuracy.ToString("F4"));

            Console.WriteLine("\nEnd neural network classification demo\n");
            Console.ReadKey();
        }
    }
}
