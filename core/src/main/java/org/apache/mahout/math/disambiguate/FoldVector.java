package org.apache.mahout.math.disambiguate;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;

import org.apache.hadoop.io.Text;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.*;

import java.io.IOException;

/**
 * @Author: Chris Hokamp
 *
 * Fold a vector into a reduced space by multiplying with SIGMA^(-1) and V_t
 * TODO: output should be saved to disk
 * TODO: the matrix loading should be able to work in distributed mode, otherwise this will break when scaled-up
 */
public class FoldVector {
    /**
     * From command line:
     * (String)eigenValueVectorPath
     * (String)columnsTransposePath
     * (int)rows
     * (int)columns
     * (String)input
     *
     */

    //Working notes:
    //
   // input directory of the documents in {@link org.apache.hadoop.io.SequenceFile} format

    public static void main(String [] args) throws IOException
    {
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);

        String eigenValueVectorPath = args[0];
        String columnsTransposePath = args[1];

        //We'll need to know the row and column sizes of V_t (the transpose of the V matrix ouput by ssvd)
        int rows = Integer.parseInt(args[2]);
        int columns = Integer.parseInt(args[3]);

        //The path to the vectors we want to fold into the space - i.e. String inputPath = "/home/chris/data/mahout_data/seq2sparse_output/tfidf-vectors/part-r-00000";
        String inputPath = args[4];

        Path eigenPath = new Path(eigenValueVectorPath);
        Path transposePath = new Path(columnsTransposePath);

        SequenceFile.Reader reader = new SequenceFile.Reader(fs, eigenPath, conf);
        DenseVector vect = new DenseVector();

        IntWritable key = new IntWritable();
        VectorWritable value = new VectorWritable();
        //TODO: this breaks if it is passed a file with more than one vector, but the reason may not be obvious - give a meaningful error message
        while (reader.next(key, value)) {
            vect = (DenseVector)value.get();
            //take the reciprocal of each value so that we end up with the inverse
            for (Vector.Element e : vect) {
                Double cellValue = (Double)e.get();
                Double reciprocal = 1/cellValue;
                e.set(reciprocal);
            }
        }
        reader.close();
        //now build the diagonal matrix
        DiagonalMatrix sigmaInverse = new DiagonalMatrix(vect);

        //now load the transpose of V
        SequenceFile.Reader matrixReader = new SequenceFile.Reader(fs, transposePath, conf);
        IntWritable matrixKey = new IntWritable();
        VectorWritable matrixValue = new VectorWritable();

        DenseMatrix transposeRight = new DenseMatrix(rows, columns);
        int rowIndex = 0;
        while (matrixReader.next(matrixKey, matrixValue))
        {
            SequentialAccessSparseVector matrixRow = (SequentialAccessSparseVector)matrixValue.get();

            transposeRight.assignRow(rowIndex, matrixRow);
            rowIndex++;
        }

        // now load the input that we want to fold into the space
        Path tfidfPath = new Path(inputPath);
        SequenceFile.Reader tfIdfReader = new SequenceFile.Reader(fs, tfidfPath, conf);
        Text vectorKey = new Text();
        VectorWritable vectorValue = new VectorWritable();

        Matrix product = sigmaInverse.times(transposeRight);

        while (tfIdfReader.next(vectorKey, vectorValue)) {
            RandomAccessSparseVector namedVector = (RandomAccessSparseVector)vectorValue.get();

            //We want to multiply this vector by the product of Sigma and V_t
            Vector folded = product.times(namedVector);

            //TEST
            int i = 0;
            for( Vector.Element e : folded ){
                System.out.println ("Initial Test 2: token " + i +": " + e.get());
                //System.out.println("Token: "+dictionaryMap.get(e.index())+", TF-IDF weight: "+e.get()) ;
                i++;
            }
            //END TEST
        }
        tfIdfReader.close();
    }
}
