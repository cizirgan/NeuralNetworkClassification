using System.IO;

namespace Haberman
{
    public static class Operations
    {
        public static void MakeTrainAndTest(string file, out double[][] trainMatrix, out double[][] testMatrix)
        {
            int numLines = 0;
            FileStream ifs = new FileStream(file, FileMode.Open);
            StreamReader sr = new StreamReader(ifs);
            while (sr.ReadLine() != null)
                ++numLines;
            sr.Close(); ifs.Close();

            int numTrain = (int)(0.80 * numLines);
            int numTest = numLines - numTrain;

            double[][] allData = new double[numLines][];  // could use Helpers.MakeMatrix here
            for (int i = 0; i < allData.Length; ++i)
                allData[i] = new double[5];               // (x0, x1, x2), (y0, y1)

            string line = "";
            string[] tokens = null;
            ifs = new FileStream(file, FileMode.Open);
            sr = new StreamReader(ifs);
            int row = 0;
            while ((line = sr.ReadLine()) != null)
            {
                tokens = line.Split(',');
                allData[row][0] = double.Parse(tokens[0]);
                allData[row][1] = double.Parse(tokens[1]);
                allData[row][2] = double.Parse(tokens[2]);


                /* for (int i = 0; i < 3; ++i)
                     allData[row][i] = 0.25 * allData[row][i] - 1.25; // scale input data to [-1.0, +1.0]*/

                allData[row][0] = 0.25 * (allData[row][0] / 10) - 1.25;
                allData[row][1] = 0.25 * (allData[row][1] / 10) - 1.25;
                allData[row][2] = 0.25 * (allData[row][2] / 10) - 1.25;


                if (tokens[3] == "1") { allData[row][3] = 1.0; allData[row][4] = 0.0; }
                else if (tokens[3] == "2") { allData[row][3] = 0.0; allData[row][4] = 1.0; }
                ++row;
            }
            sr.Close(); ifs.Close();


            Helpers.CreateHaberData("../ok.txt", allData);

            Helpers.ShuffleRows(allData);

            trainMatrix = Helpers.MakeMatrix(numTrain, 5);
            testMatrix = Helpers.MakeMatrix(numTest, 5);

            for (int i = 0; i < numTrain; ++i)
            {
                allData[i].CopyTo(trainMatrix[i], 0);
            }

            for (int i = 0; i < numTest; ++i)
            {
                allData[i + numTrain].CopyTo(testMatrix[i], 0);
            }
        }

    }
}