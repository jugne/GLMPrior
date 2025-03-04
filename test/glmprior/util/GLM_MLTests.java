package glmprior.util;

import beast.base.inference.parameter.RealParameter;
import beast.base.inference.parameter.IntegerParameter;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class GLM_MLTests {

    private GLM_ML glmML;
    private ArrayList<RealParameter> predictors;
    private ArrayList<RealParameter> weights;
    private IntegerParameter nOutputs;
    private Integer layers;
    private List<Integer> nodes;

    @BeforeEach
    void setUp() {
        // Define predictors
        predictors = new ArrayList<>();
        // nPredictors=2, nInstances=3
        predictors.add(new RealParameter("0.5 1.0 1.5"));
        predictors.add(new RealParameter("2.0 2.5 3.0"));

        // Define weights
        weights = new ArrayList<>();
        weights.add(new RealParameter("0.1 0.2 0.3 0.4")); // For first layer (nPredictors x nNodes)
        weights.add(new RealParameter("0.7 0.8")); // Output layer

        // Define outputs and layers
        nOutputs = new IntegerParameter("1");
        layers = 1; // One hidden layer
        nodes = Arrays.asList(2); // One hidden layer with 2 nodes

        // Initialize GLM_ML instance
        glmML = new GLM_ML();
        glmML.predictorsInput.setValue(predictors, glmML);
        glmML.weightsInput.setValue(weights, glmML);
        glmML.nOutputsInput.setValue(nOutputs, glmML);
        glmML.layersInput.setValue(layers, glmML);
        glmML.nodesInput.setValue(nodes, glmML);

        glmML.initAndValidate();
    }

    @Test
    void testInitializationValidInputs() {
        assertNotNull(glmML);
        assertEquals(2, glmML.predictorsInput.get().size());
        assertEquals(1, glmML.nOutputsInput.get().getValue());
        assertEquals(1, glmML.layersInput.get());
        assertEquals(1, glmML.nodesInput.get().size());
        assertEquals(2, glmML.nodesInput.get().get(0));
    }

    @Test
    void testInvalidPredictorDimension() {
        predictors.add(new RealParameter("4.0")); // Mismatched dimension
        glmML.predictorsInput.setValue(predictors, glmML);
        assertThrows(IllegalArgumentException.class, glmML::initAndValidate);
    }

    @Test
    void testInvalidNodesLayerMismatch() {
        glmML.layersInput.setValue(2, glmML);
        assertThrows(IllegalArgumentException.class, glmML::initAndValidate);
    }

    @Test
    void testGetDimension() {
        assertEquals(3, glmML.getDimension());
    }

    @Test
    void testGetArrayValue() {
        for (int i = 0; i < glmML.getDimension(); i++){
            assertNotNull(glmML.getArrayValue(i));
            System.out.print(glmML.getArrayValue(i) + " ");
        }
    }
}
