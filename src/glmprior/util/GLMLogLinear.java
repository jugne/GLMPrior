package glmprior.util;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.inference.CalculationNode;
import beast.base.core.Function;

/**
 * @author Cecilia Valenzuela Agui
 */
@Description(
        "A function that implements a GLM Log Linear model from a set of predictors, "
                + "coefficients, indicator variables for predictor selection, and optionally "
                + "a global scale factor and error terms."
)
public class GLMLogLinear extends CalculationNode implements Function {

    public Input<Function> predictorsInput = new Input<>("predictors",
            "GLM List of predictors, e.g. numbers of flights between different locations",
            Input.Validate.REQUIRED);

    public Input<Function> coefficientsInput = new Input<>("coefficients",
            "GLM coefficients.", Input.Validate.REQUIRED);

    public Input<Function> indicatorsInput = new Input<>("indicators",
            "Indicators for predictor inclusion/exclusion in GLM.", Input.Validate.REQUIRED);

    public Input<Function> scaleFactorInput = new Input<>("scaleFactor",
            "Scale factor.", Input.Validate.OPTIONAL);

    public Input<Function> errorInput = new Input<>("error",
            "Error terms.", Input.Validate.OPTIONAL);

    Function predictors, coefficients, indicators, scaleFactor, error;
    int predictorSize, coefficientsSize;

    @Override
    public void initAndValidate() {
        predictors = predictorsInput.get();
        coefficients = coefficientsInput.get();
        indicators = indicatorsInput.get();

        scaleFactor = scaleFactorInput.get();
        error = errorInput.get();

        if (predictors.getDimension() % coefficients.getDimension() != 0)
            throw new IllegalArgumentException("GLM Predictor list dimension is not a multiple "
                    + "of the number of GLM coefficients.");

        predictorSize = predictors.getDimension()/(coefficients.getDimension());
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
            if (error.getDimension() != predictorSize)
                throw new IllegalArgumentException("GLM error term has an incorrect number "
                + "of elements. It should be equal to parameter dimension.");
        }
    }

    @Override
    public int getDimension() {
        return predictorSize;
    }

    @Override
    public double getArrayValue(int i) {
        double lograte = 0;
        for (int j = 0; j < coefficients.getDimension(); j++) {
            if (indicators.getArrayValue(j) > 0.0) {
                lograte += coefficients.getArrayValue(j) * predictors.getArrayValue(j * predictorSize + i);
            }
        }

        if (error != null)
            lograte += error.getArrayValue(i);

        return scaleFactor.getArrayValue() * Math.exp(lograte);
    }
}