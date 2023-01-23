package glmprior.util;

import beast.base.core.Function;
import beast.base.inference.parameter.BooleanParameter;
import beast.base.inference.parameter.RealParameter;
import org.junit.Assert;
import org.junit.Test;

public class GLMLogLinearTest {

    @Test
    public void testGLM() {
        Function pred = new RealParameter("1.0 2.0 3.0 4.0");
        Function coeff = new RealParameter("0.3 0.5");
        Function ind = new BooleanParameter("1 1");
        Function scale = new RealParameter("2.0");

        GLMLogLinear glm = new GLMLogLinear();
        glm.initByName(
                "predictors", pred,
                "coefficients", coeff,
                "indicators", ind,
                "scaleFactor", scale);

        for (int i = 0; i < glm.getDimension(); i++)
            System.out.print(glm.getArrayValue(i) + " ");

        Assert.assertEquals(2, glm.getDimension());

        Assert.assertEquals(2.0 * Math.exp(1.8), glm.getArrayValue(0), 1e-10);
        Assert.assertEquals(2.0 * Math.exp(2.6), glm.getArrayValue(1), 1e-10);
    }

    @Test
    public void testGLMind0() {
        Function pred = new RealParameter("1.0 2.0 3.0 4.0");
        Function coeff = new RealParameter("0.3 0.5");
        Function ind = new BooleanParameter("0 0");
        Function scale = new RealParameter("2.0");

        GLMLogLinear glm = new GLMLogLinear();
        glm.initByName(
                "predictors", pred,
                "coefficients", coeff,
                "indicators", ind,
                "scaleFactor", scale);

        for (int i = 0; i < glm.getDimension(); i++)
            System.out.print(glm.getArrayValue(i) + " ");

        Assert.assertEquals(2, glm.getDimension());

        Assert.assertEquals(2.0, glm.getArrayValue(0), 1e-10);
        Assert.assertEquals(2.0, glm.getArrayValue(1), 1e-10);
    }

    @Test
    public void testGLMerror() {
        Function pred = new RealParameter("1.0 2.0 3.0 4.0");
        Function coeff = new RealParameter("0.3 0.5");
        Function ind = new BooleanParameter("1 1");
        Function scale = new RealParameter("2.0");
        Function err = new RealParameter("0.1 0.2");

        GLMLogLinear glm = new GLMLogLinear();
        glm.initByName(
                "predictors", pred,
                "coefficients", coeff,
                "indicators", ind,
                "scaleFactor", scale,
                "error", err);

        for (int i = 0; i < glm.getDimension(); i++)
            System.out.print(glm.getArrayValue(i) + " ");

        Assert.assertEquals(2, glm.getDimension());

        Assert.assertEquals(2.0 * Math.exp(1.9), glm.getArrayValue(0), 1e-10);
        Assert.assertEquals(2.0 * Math.exp(2.8), glm.getArrayValue(1), 1e-10);
    }
}