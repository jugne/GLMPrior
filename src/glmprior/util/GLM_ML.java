package glmprior.util;

import beast.base.core.Description;
import beast.base.core.Function;
import beast.base.core.Input;
import beast.base.inference.CalculationNode;
import beast.base.inference.parameter.IntegerParameter;
import beast.base.inference.parameter.RealParameter;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;


@Description("Designed to be used within Beast2." +
        "The weights are the parameters to be estimated. Training of weights should be implemented" +
        "The predictors are the input data. The number of outputs is the number of outputs in the output layer. " +
        "The number of layers is the number of hidden layers. " +
        "The nodes are the number of nodes in each hidden layer." +
        "The hidden layers use the ReLU activation function. The output layer uses the softplus activation function" )

public class GLM_ML extends CalculationNode implements Function {

    public  Input<ArrayList<RealParameter>>  predictorsInput = new Input<>("predictor",
            "One or more predictor for the GLM, e.g. numbers of flights between different locations",
            new ArrayList<>(), Input.Validate.REQUIRED);

    public Input<ArrayList<RealParameter>> weightsInput = new Input<>("weights",
            "GLM_ML weights for each layer (hidden and output).",new ArrayList<>(), Input.Validate.REQUIRED);

    public Input<IntegerParameter> nOutputsInput = new Input<>("nOutputs",
            "GLM_ML number of outputs the output layer. Default is 1.");
    public Input<Integer> layersInput = new Input<>("layers",
            "Number of hidden layers in GLM_ML. Default is 1.", 1);

    public Input<List<Integer>> nodesInput = new Input<>("nodes",
            "Number of nodes in each hidden layer in GLM_ML.", new ArrayList<>());



    // TODO Maybe have some of the below in the future
    // Different activation functions?
//    public Input<BooleanParameter> indicatorsInput = new Input<>("indicators",
//            "Indicators for predictor inclusion/exclusion in GLM.", Input.Validate.REQUIRED);
//    public Input<RealParameter> scaleFactorInput = new Input<>("scaleFactor",
//            "Scale factor.", new RealParameter("1.0"), Input.Validate.OPTIONAL);
//
//    public Input<RealParameter> errorInput = new Input<>("error",
//            "Error terms.", Input.Validate.OPTIONAL);
//
//    public Input<Boolean> transformInput = new Input<>("transform",
//            "Boolean value to log transform and scale predictors. Default true.", false,
//            Input.Validate.OPTIONAL);


    INDArray predictors;
    INDArray inputData;
    INDArray output;
    ArrayList<RealParameter> weights;
    int parameterSize, nPredictor; // parameter size as instances in the original example
    Integer[] nWeights;
    int nOutputs;
    int nLayers;

    @Override
    public void initAndValidate() {


        nPredictor = predictorsInput.get().size();
        parameterSize = predictorsInput.get().get(0).getDimension();
        for (RealParameter pred : predictorsInput.get())
            if (parameterSize != pred.getDimension())
                throw new IllegalArgumentException("GLM Predictors do not have the same dimension " +
                        parameterSize + "!=" +  pred.getDimension());

        List<Integer> nodes = nodesInput.get();
        nLayers = layersInput.get();
        if (nodes.size() != nLayers)
            throw new IllegalArgumentException("GLM number of nodes do not have the same dimension as number of layers " +
                    nodes.size() + "!=" + nLayers);


        // Convert predictor input to DoubleArray, make ND4J matrix and transpose
        predictors = Nd4j.create(convertToDoubleArray(predictorsInput.get())).transpose();
        weights = weightsInput.get();
        nWeights = new Integer[weights.size()];
        nOutputs = nOutputsInput.get().getValue();



        if (nLayers == 0) {
            weights.get(0).setDimension(nPredictor * nOutputs);
        } else {
            for (int i = 0; i < nLayers; i++) {
                if (i == 0) {
                    nWeights[i] = nPredictor * nodes.get(0);
                } else {
                    nWeights[i] = nodes.get(i - 1) * nodes.get(i);
                }
                weights.get(i).setDimension(nWeights[i]);
            }
            weights.get(nLayers).setDimension(nodes.get(nLayers - 1) * nOutputs);
        }

    }

    @Override
    public int getDimension() {
        return parameterSize;
    }

    @Override
    public double getArrayValue(int i) {
        for (RealParameter w : weights){
            if (w.somethingIsDirty()){ // recalculate if new weights were proposed
                recalculate();
            }
        }
        return output.getScalar(i).getDouble();
    }

    private void recalculate() {
        if (nLayers == 0) {
            // Direct linear transformation without hidden layers
            INDArray w = Nd4j.create(weights.get(0).getDoubleValues()).reshape(nPredictor, nOutputs);
            output = predictors.mmul(w);
        } else {
            // Hidden layers
            for (int l = 0; l < nLayers; l++) {
                if (l == 0) {
                    INDArray w = Nd4j.create(weights.get(0).getDoubleValues()).reshape(nPredictor, nodesInput.get().get(0));
                    output = runLayer(predictors, w, GLM_ML::relu);
                } else {
                    INDArray w = Nd4j.create(weights.get(l).getDoubleValues()).reshape(nodesInput.get().get(l - 1), nodesInput.get().get(l));
                    output = runLayer(output, w, GLM_ML::relu);
                }
            }
            // Output layer
            output = runLayer(output, Nd4j.create(weights.get(nLayers).getDoubleValues()).reshape(nodesInput.get().get(nLayers - 1), nOutputs), GLM_ML::softplus);
        }
    }



    // TODO could also just use activation functions from:
    // https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/activations/Activation.java

    private static INDArray softplus(INDArray z) {
        return Transforms.log(Transforms.exp(z).add(1), false);
    }

    private static INDArray relu(INDArray z) {
        return Transforms.max(z, 0);
    }

    private static INDArray runLayer(INDArray x1, INDArray x2, java.util.function.Function<INDArray, INDArray> activationFunction) {
        INDArray z = x1.mmul(x2);
        if (activationFunction != null) {
            z = activationFunction.apply(z);
        }
        return z;
    }


    // Helper converter
    private static double[][] convertToDoubleArray(ArrayList<RealParameter> realParams) {
        // Dimensions:
        int rows = realParams.size();
        int cols = realParams.get(0).getDimension(); // All must have the same dimension

        double[][] result = new double[rows][cols];

        // Fill the 2D array
        for (int i = 0; i < rows; i++) {
            RealParameter rp = realParams.get(i);
            for (int j = 0; j < cols; j++) {
                result[i][j] = rp.getValue(j);
            }
        }

        return result;
    }
}