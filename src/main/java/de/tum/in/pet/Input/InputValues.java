package de.tum.in.pet.Input;

import de.tum.in.pet.implementation.meanPayoff.DeltaTCalculationMethod;
import de.tum.in.pet.implementation.meanPayoff.SimulateMec;
import de.tum.in.pet.implementation.reachability.UpdateMethod;
import de.tum.in.pet.sampler.SuccessorHeuristic;
import de.tum.in.probmodels.explorer.InformationLevel;

public class InputValues {
    public final double precision;
    public final int revisitThreshold;
    public final double maxReward;
    public final double pMin;
    public final double errorTolerance;
    public final int iterSamples;
    public final long timeout;
    public final boolean getErrorProbability;
    public final SuccessorHeuristic successorHeuristic;
    public final InformationLevel informationLevel;
    public final UpdateMethod updateMethod;
    public final String rewardStructure;
    public final boolean solveUsingQP;
    public final SimulateMec simulateMec;
    public final String outputPath;
    public final int maxSuccessorsInModel;
    public final DeltaTCalculationMethod deltaTCalculationMethod;
    public final double alpha;


    public InputValues(double precision, int revisitThreshold, double maxReward, double pMin, double errorTolerance,
                       int iterSamples, long timeout, boolean getErrorProbability, SuccessorHeuristic successorHeuristic,
                       InformationLevel informationLevel, UpdateMethod updateMethod, String rewardStructure, boolean solveUsingQP,
                       SimulateMec simulateMec, String outputPath, int maxSuccessorsInModel, DeltaTCalculationMethod deltaTCalculationMethod, double alpha) {
        this.precision = precision;
        this.revisitThreshold = revisitThreshold;
        this.maxReward = maxReward;
        this.pMin = pMin;
        this.errorTolerance = errorTolerance;
        this.iterSamples = iterSamples;
        this.timeout = timeout;
        this.getErrorProbability = getErrorProbability;
        this.successorHeuristic = successorHeuristic;
        this.informationLevel = informationLevel;
        this.updateMethod = updateMethod;
        this.rewardStructure = rewardStructure;
        this.solveUsingQP = solveUsingQP;
        this.simulateMec = simulateMec;
        this.outputPath = outputPath;
        this.maxSuccessorsInModel = maxSuccessorsInModel;
        this.deltaTCalculationMethod = deltaTCalculationMethod;
        this.alpha = alpha;
    }
}
