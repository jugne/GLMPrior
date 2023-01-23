package glmprior.util;

import beast.base.core.Description;
import beast.base.inference.CalculationNode;
import beast.base.core.Function;
import beast.base.inference.distribution.ParametricDistribution;
import org.apache.commons.math.distribution.Distribution;

@Description(
        "GLM dummy class for beauti integration."
)
public class GLMPrior extends ParametricDistribution {

    @Override
    public void initAndValidate() {
    }

    @Override
    public Distribution getDistribution() {
        return null;
    }
}