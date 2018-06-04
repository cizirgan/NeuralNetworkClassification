using System;
using System.IO;

namespace Haberman
{
    public static class Helpers
    {
        public static void MakeData(string dataFile, int numLines, Random rnd)
        {
            double[] weights = new double[] { -0.1, 0.2, -0.3, 0.4, -0.5,
                                        0.6, -0.7, 0.8, -0.9, 1.0,
                                        -1.1, 1.2, -1.3, 1.4, -1.5,
                                        1.6, -1.7, 1.8, -1.9, 2.0,
                                        -0.5, 0.6, -0.7, 0.8, -0.9,
                                        1.5, -1.4, 1.3,
                                        -1.2, 1.1, -1.0,
                                        0.9, -0.8, 0.7,
                                        -0.6, 0.5, -0.4,
                                        0.3, -0.2, 0.1,
                                        0.1, -0.3, 0.6 };

            NeuralNetwork nn = new NeuralNetwork(4, 5, 3);
            nn.SetWeights(weights);

            FileStream ofs = new FileStream(dataFile, FileMode.Create);
            StreamWriter sw = new StreamWriter(ofs);

            for (int i = 0; i < numLines; ++i)
            {
                double[] inputs = new double[4];
                for (int j = 0; j < inputs.Length; ++j)
                    inputs[j] = rnd.Next(1, 10);

                double[] outputs = nn.ComputeOutputs(inputs);

                string color = "";
                int idx = Helpers.IndexOfLargest(outputs);
                if (idx == 0) { color = "red"; }
                else if (idx == 1) { color = "green"; }
                else if (idx == 2) { color = "blue"; }

                sw.WriteLine(inputs[0].ToString("F1") + " " + inputs[1].ToString("F1") + " " + inputs[2].ToString("F1") + " " + inputs[3].ToString("F1") + " " + color);
            }
            sw.Close(); ofs.Close();

        }

        static Random rnd = new Random(0);

        public static double[][] MakeMatrix(int rows, int cols)
        {
            double[][] result = new double[rows][];
            for (int i = 0; i < rows; ++i)
                result[i] = new double[cols];
            return result;
        }

        public static void ShuffleRows(double[][] matrix)
        {
            for (int i = 0; i < matrix.Length; ++i)
            {
                int r = rnd.Next(i, matrix.Length);
                double[] tmp = matrix[r];
                matrix[r] = matrix[i];
                matrix[i] = tmp;
            }
        }

        public static int IndexOfLargest(double[] vector)
        {
            int indexOfLargest = 0;
            double maxVal = vector[0];
            for (int i = 0; i < vector.Length; ++i)
            {
                if (vector[i] > maxVal)
                {
                    maxVal = vector[i];
                    indexOfLargest = i;
                }
            }
            return indexOfLargest;
        }

        public static void ShowVector(double[] vector, int decimals, bool newLine)
        {
            string fmt = "F" + decimals;
            for (int i = 0; i < vector.Length; ++i)
            {
                if (i > 0 && i % 12 == 0)
                    Console.WriteLine("");
                if (vector[i] >= 0.0) Console.Write(" ");
                Console.Write(vector[i].ToString(fmt) + " ");
            }
            if (newLine == true) Console.WriteLine("");
        }

        public static void ShowMatrix(double[][] matrix, int numRows)
        {
            int ct = 0;
            if (numRows == -1) numRows = int.MaxValue;
            for (int i = 0; i < matrix.Length && ct < numRows; ++i)
            {
                for (int j = 0; j < matrix[0].Length; ++j)
                {
                    if (matrix[i][j] >= 0.0) Console.Write(" ");
                    if (j == 4) Console.Write("-> ");
                    Console.Write(matrix[i][j].ToString("F2") + " ");
                }
                Console.WriteLine("");
                ++ct;
            }
            Console.WriteLine("");
        }

        public static void ShowTextFile(string textFile, int numLines)
        {
            FileStream ifs = new FileStream(textFile, FileMode.Open);
            StreamReader sr = new StreamReader(ifs);
            string line = "";
            int ct = 0;
            while ((line = sr.ReadLine()) != null && ct < numLines)
            {
                Console.WriteLine(line);
                ++ct;
            }
            sr.Close(); ifs.Close();
        }

        public static void CreateHaberData(string dataFile, double[][] allData)
        {
            FileStream ofs = new FileStream(dataFile, FileMode.Create);
            StreamWriter sw = new StreamWriter(ofs);
            var numberOfRows = allData.GetLength(0);
            var numberOfColumns = allData[0].Length;

            using (sw)
            {
                for (int row = 0; row < numberOfRows; row++)
                {
                    for (int col = 0; col < numberOfColumns; col++)
                    {
                        sw.Write(allData[row][col].ToString("F2") + " ");
                    }
                    sw.WriteLine();
                }
            }
        sw.Close(); ofs.Close();
        }
    }
}