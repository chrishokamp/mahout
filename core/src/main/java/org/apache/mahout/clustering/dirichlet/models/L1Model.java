package org.apache.mahout.clustering.dirichlet.models;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Iterator;

import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.Vector.Element;

public class L1Model implements Model<VectorWritable> {

  private static DistanceMeasure measure = new ManhattanDistanceMeasure();

  public L1Model() {
    super();
  }

  public L1Model(Vector v) {
    observed = v.like();
    coefficients = v;
  }

  private Vector coefficients;

  private int count = 0;

  private Vector observed;

  @Override
  public void computeParameters() {
    coefficients = observed.divide(count);
  }

  @Override
  public int count() {
    return count;
  }

  @Override
  public void observe(VectorWritable x) {
    count++;
    x.get().addTo(observed);
  }

  @Override
  public double pdf(VectorWritable x) {
    return Math.exp(-measure.distance(x.get(), coefficients));
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    coefficients = VectorWritable.readVector(in);
  }

  @Override
  public void write(DataOutput out) throws IOException {
    VectorWritable.writeVector(out, coefficients);
  }

  public L1Model sample() {
    return new L1Model(coefficients.clone());
  }

  @Override
  public String toString() {
    StringBuilder buf = new StringBuilder();
    buf.append("l1m{n=").append(count).append(" c=[");
    int nextIx = 0;
    if (coefficients != null) {
      // handle sparse Vectors gracefully, suppressing zero values
      for (int i = 0; i < coefficients.size(); i++) {
        double elem = coefficients.get(i);
        if (elem == 0.0)
          continue;
        if (i > nextIx)
          buf.append("..{").append(i).append("}=");
        buf.append(String.format("%.2f", elem)).append(", ");
        nextIx = i + 1;
      }
    }
    buf.append("]}");
    return buf.toString();
  }

}