package glmprior.util;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.inference.CalculationNode;
import beast.base.core.Function;
import beast.base.inference.parameter.RealParameter;

import java.util.ArrayList;
import java.util.List;


/**
 * @author Cecilia Valenzuela Agui
 */
@Description(
        "A function that implements a GLM Log Linear model from a set of predictors, "
                + "coefficients, indicator variables for predictor selection, and optionally "
                + "a global scale factor and error terms. Predictors are log transform and scale by default."
)
public class GLMLogLinear extends CalculationNode implements Function {

    public  Input<List<Function>>  predictorsInput = new Input<>("predictor",
            "One or more predictor for the GLM, e.g. numbers of flights between different locations",
            new ArrayList<>(), Input.Validate.REQUIRED);

    public Input<Function> coefficientsInput = new Input<>("coefficients",
            "GLM coefficients.", Input.Validate.REQUIRED);

    public Input<Function> indicatorsInput = new Input<>("indicators",
            "Indicators for predictor inclusion/exclusion in GLM.", Input.Validate.REQUIRED);

    public Input<Function> scaleFactorInput = new Input<>("scaleFactor",
            "Scale factor.", Input.Validate.OPTIONAL);

    public Input<Function> errorInput = new Input<>("error",
            "Error terms.", Input.Validate.OPTIONAL);

    public Input<Boolean> transformInput = new Input<>("transform",
            "Boolean value to log transform and scale predictors. Default true.", true,
            Input.Validate.OPTIONAL);

    public  Input<List<Function>>  predictorsTInput = new Input<>("predictorT",
            "Predictor transformed, internal to the class",
            new ArrayList<>(), Input.Validate.OPTIONAL);

    List<Function> predictors;
    Function coefficients, indicators, scaleFactor, error;
    int parameterSize, coefficientsSize, predictorsSize;

    @Override
    public void initAndValidate() {
        coefficients = coefficientsInput.get();
        indicators = indicatorsInput.get();
        scaleFactor = scaleFactorInput.get();
        predictors = predictorsInput.get();

        predictorsSize = 0;
        for (Function pred : predictors)
            predictorsSize += pred.getDimension();

        if (predictorsSize % coefficients.getDimension() != 0)
            throw new IllegalArgumentException("GLM Predictor list dimension is not a multiple "
                    + "of the number of GLM coefficients.");

        parameterSize = predictorsSize/(coefficients.getDimension());
        coefficientsSize = coefficients.getDimension();

        if (indicators.getDimension() != coefficientsSize)
            throw new IllegalArgumentException("GLM Coefficients and indicators "
                    + "do not have the same number of elements.");

        if (indicators.getDimension() != coefficientsSize)
            throw new IllegalArgumentException("Dimension of GLM scale factor should be 1.");

        for (int i = 0; i < indicators.getDimension(); i++) {
            if (!(indicators.getArrayValue(i) != 0 || indicators.getArrayValue(i) != 1))
                throw new IllegalArgumentException("GLM indicators incorrect value, it should be 0 or 1.");
        }

        if (errorInput.get() != null) {
            error = errorInput.get();
            if (error.getDimension() != parameterSize)
                throw new IllegalArgumentException("GLM error term has an incorrect number "
                + "of elements. It should be equal to parameter dimension.");
        }

        if (transformInput.get()) {
            Double[] predT;
            double pred, mean, sd;
            for (int j = 0; j < coefficientsSize; j++) {
                predT = new Double[parameterSize];
                mean = 0;
                sd = 0;
                for (int i = 0; i < parameterSize; i++) {
                    pred = predictorsInput.get().get(j).getArrayValue(i);
                    if (pred < 0.0)
                        throw new IllegalArgumentException("Predictor should not be smaller than 0 to be log transformed.");
                    predT[i] = Math.log(pred + 1);
                    mean += predT[i];
                }
                mean /= parameterSize;
                for (int i = 0; i < parameterSize; i++) {
                    sd += (predT[i] - mean) * (predT[i] - mean);
                }
                sd = Math.sqrt(sd / (parameterSize));
                for (int i = 0; i < parameterSize; i++) {
                    predT[i] = (predT[i] - mean) / sd;
                }
                Function fpredT = new RealParameter(predT);
                predictorsTInput.setValue(fpredT, this);
            }
            predictors = predictorsTInput.get();
        }
//        Check that predictor values can be log scaled, meaning no values that are 0 or smaller
//        Set predictors input to list double, or find a way to set values from doubles to input

    }

    @Override
    public int getDimension() {
        return parameterSize;
    }

    @Override
    public double getArrayValue(int i) {
        double lograte = 0;
        for (int j = 0; j < coefficients.getDimension(); j++) {
            if (indicators.getArrayValue(j) > 0.0) {
                lograte += coefficients.getArrayValue(j) * predictors.get(j).getArrayValue(i);
            }
        }

        if (error != null)
            lograte += error.getArrayValue(i);

        return scaleFactor.getArrayValue() * Math.exp(lograte);
    }
}