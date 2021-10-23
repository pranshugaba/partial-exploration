package de.tum.in.pet.implementation.meanPayoff;

import de.tum.in.naturals.set.NatBitSet;
import de.tum.in.naturals.set.NatBitSets;
import de.tum.in.pet.implementation.reachability.GreyUnboundedReachValues;
import de.tum.in.pet.sampler.UnboundedValues;
import de.tum.in.pet.values.Bounds;
import de.tum.in.probmodels.explorer.Explorer;
import de.tum.in.probmodels.explorer.GreyExplorer;
import de.tum.in.probmodels.generator.RewardGenerator;
import de.tum.in.probmodels.graph.Mec;
import de.tum.in.probmodels.model.Distribution;
import de.tum.in.probmodels.model.Model;
import it.unimi.dsi.fastutil.doubles.Double2LongFunction;
import it.unimi.dsi.fastutil.ints.*;
import prism.Pair;
import prism.PrismException;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static de.tum.in.probmodels.util.Util.isZero;


public class GreyOnDemandValueIterator<S, M extends Model> extends OnDemandValueIterator<S, M>{

    private final double pMin; // as mentioned in CAV'19. It should be set to the lowest transition probability of the input model.
    private final double errorTolerance; // as mentioned in CAV'19. Error tolerance for the learned distributions of the learned model.
    private final Double2LongFunction nSampleFunction; // returns N_k for each k as in CAV'19. Returns the number of times paths should be sampled for each value of k.

    private List<NatBitSet> mecs = new ArrayList<>(); // Holds a list of mecs in the model.
    private Double transDelta = 1d; // equal to delta_T as mentioned in CAV'19. Error tolerance for each transition of the learned model.

    private final Int2IntMap stateToMecMap = new Int2IntOpenHashMap(); // Map that returns the mec Index the state is a part of.
    private Int2ObjectMap<Distribution> stayActionMap = new Int2ObjectOpenHashMap<>(); // Map that holds the stay action for mecs, accessible using mecIndices.

    private Int2IntMap stayActionCounts = new Int2IntOpenHashMap(); // Map that holds the number of times each stay action for an mec has been sampled, accessible using mecIndices.

    public GreyOnDemandValueIterator(Explorer<S, M> explorer, UnboundedValues values, RewardGenerator<S> rewardGenerator,
                                     int revisitThreshold, double rMax, double pMin, double errorTolerance,
                                     Double2LongFunction nSampleFunction, double precision, long timeout) {
        super(explorer, values, rewardGenerator, revisitThreshold, rMax, precision, timeout);
        this.pMin = pMin;
        this.errorTolerance = errorTolerance;
        this.nSampleFunction = nSampleFunction;

        initGreyUnboundedReachValues();
    }

    private void initGreyUnboundedReachValues() {
        GreyUnboundedReachValues greyUnboundedReachValues = (GreyUnboundedReachValues) this.values;

        GreyExplorer<S, M> greyExplorer = (GreyExplorer<S, M>) this.explorer;

        greyUnboundedReachValues.setChoiceFunction(this::choices);
        Int2ObjectFunction<Int2IntFunction> getNumSuccessors = x -> (y -> greyExplorer.getOriginalNumOfSuccessors(x, y));
        greyUnboundedReachValues.setGetNumSuccessors(getNumSuccessors);
        greyUnboundedReachValues.setIsInMEC(stateToMecMap::containsKey);
    }

    @Override
    public Bounds bounds(int state) {
        return values.bounds(state);
    }

    @Override
    protected boolean sample(int initialState, int run) throws PrismException {

        GreyUnboundedReachValues values = (GreyUnboundedReachValues) this.values;

        GreyExplorer<S, M> explorer = (GreyExplorer<S, M>) explorer();

        double k = Math.pow(2, run);
        long nIterations = nSampleFunction.apply(k);
        double errorTolerance = this.errorTolerance;

        Random randomIntegerSampler = new Random();

        // Updates the confidenceWidthFunction according to the latest counts and transDelta value. The confidenceWidthFunction
        // returns the confidenceWidth for a state x and an action with index y. if y is greater than the number of choices
        // the explorer holds, it must be the stay action. We set confidence width of stay action equal to zero as we
        // know the probabilities of the action are accurate as they have been calculated and not learned.
        Int2ObjectFunction<Int2DoubleFunction> confidenceWidthFunction = x -> (y -> y < explorer.getChoices(x).size()
                ? Math.sqrt(-Math.log(transDelta)/(2*explorer.getActionCounts(x, y)))
                : 0);

        // Updates the confidence width function in UnboundedReachValues.
        values.setConfidenceWidthFunction(confidenceWidthFunction);

        for (int i = 0; i < nIterations; i++) {
            int currentState = initialState;
            int prevState = initialState;
            Int2IntOpenHashMap stateVisitCounts = new Int2IntOpenHashMap();  // keeps counts of the number of times a state is visited

            while (true) {
                stateVisitCounts.putIfAbsent(currentState, 0);
                stateVisitCounts.addTo(currentState, 1);

                // checks plus state,minus state and uncertain state
                if (BoundedMecQuotient.isSinkState(currentState)) {
                    // We update the MEC reward bounds through running VI if we reach the uncertain or the plus state. This is
                    // slightly different from the version in CAV'17 where VI is only run when the uncertain state is reached.
                    // However, this is also OK as reaching the plus state shows that probably the lower reward bound is high
                    // enough, meaning the EC is promising and it is worth getting a more precise value. We make sure in updateMEC
                    // that we don't get value that is more precise than what is required.
                    if (BoundedMecQuotient.isUncertainState(currentState)||BoundedMecQuotient.isPlusState(currentState)) {
                        int mecIndex = stateToMecMap.get(prevState);
                        explorer.activateFullyExploredActionsModel();
                        updateMec(mecIndex);
                        explorer.deactivateFullyExploredActionsModel();
                    }
                    break;
                }

                if (!explorer().isExploredState(currentState)) {
                    explore(currentState);  // action choices etc. are populated in the partial model. The bounds of currentState are also initialised.
                }

                List<Distribution> choices = choices(currentState);
                if (choices.isEmpty()){
                    break;
                }

                int nextState, nextActionIndex;
                // This condition is there as in the simulate function in CAV'19. It checks whether we have been returning to a
                // state too many times during simulation indicating that we could be stuck inside an MEC.
                if (stateVisitCounts.get(currentState)>=revisitThreshold && looping(currentState)) {
                    int mecIndex = stateToMecMap.get(currentState);
                    NatBitSet mecStates = this.mecs.get(mecIndex);
                    List<Pair<Integer, Integer>> bestActionStatePairs = values.getBestLeavingAction(mecStates, this::choices);
                    int sampledActionStatePair = randomIntegerSampler.nextInt(bestActionStatePairs.size());
                    currentState = bestActionStatePairs.get(sampledActionStatePair).first;
                    nextActionIndex = bestActionStatePairs.get(sampledActionStatePair).second;
                    choices = choices(currentState);
                }
                else {
                    nextActionIndex = values.sampleNextAction(currentState, choices); // index of the action from the state that is to be sampled next.
                    // this happens when none of the actions look promising at all, i.e. all actions have a upper bound of 0.
                    // To continue the simulation, we forcefully sample an action.
                    if (nextActionIndex == -1) {
                        nextActionIndex = explorer.sampleNextAction(currentState);
                    }
                }

                assert nextActionIndex != -1;

                // If the sampled action's index is the last index and state is a part of an mec, then this index of a stay action.
                // Here, we simply sample the next state. However, if we don't have a stay action, we have to call the explorer to
                // sample the next state according to the real distributions.
                if (nextActionIndex == choices.size()-1 && stateToMecMap.containsKey(currentState)){
                    nextState = choices.get(nextActionIndex).sample();
                    stayActionCounts.put(stateToMecMap.get(currentState), stayActionCounts.get(stateToMecMap.get(currentState))+1);
                }
                else {
                    nextState = explorer.simulateAction(currentState, nextActionIndex);
                    // If this action has been sampled enough number of times, we know that it can now be considered as a part of an MEC.
                    // Hence, we know that there might be new MECs in the model and it could be worthwhile finding them again.

                    explorer.updateCounts(currentState, nextActionIndex, nextState);
                }

                // This is true when the currentState doesn't have any choices from it, i.e. it is a sink state.
                if (nextState == -1) {
                    break;
                }

                prevState = currentState;
                currentState = nextState;

                transDelta = errorTolerance*pMin/explorer.getNumTrans();

            }
        }

        handleComponents();

        values.resetBounds();
        initSinkStates();

        confidenceWidthFunction = x -> (y -> y < explorer.getChoices(x).size()
                ? Math.sqrt(-Math.log(transDelta)/(2*explorer.getActionCounts(x, y)))
                : 0);
        values.setConfidenceWidthFunction(confidenceWidthFunction);

        // the update function is ran until there has been some progress, i.e., the upper bounds of some state have been changed.
        // if there has been change, this change needs to be propagated through the rest of the states.
        boolean ifProgress = true;
        int nMaxUpdates = explorer.exploredStateCount();
        int nUpdates = 0;
        while(ifProgress && nUpdates < nMaxUpdates) {
            ifProgress = update();
            nUpdates++;
        }

        return true;

    }

    /**
     * Updates the bounds of the model according to the latest changes.
     * @return true, if there have been any changes to the bounds of the states, else false.
     */
    private boolean update(){
        GreyUnboundedReachValues values = (GreyUnboundedReachValues) this.values;
        values.cacheCurrBounds(); // cache the current bounds to check if we will make any progress in update.

        for (int state: explorer.exploredStates()){
            List<Distribution> realChoices = choices(state);
            values.update(state, realChoices);
        }

        for (NatBitSet mec : this.mecs) {
            values.deflate(mec, this::choices);
        }

        return values.checkProgress();
    }

    /**
     * @param mecIndex: index of MEC for which bounds reward bounds are desired.
     * @return Reward bounds of the MEC in question.
     */
    @Override
    protected Bounds getMecBounds(int mecIndex) {
        return BoundedMecQuotient.getBoundsFromStayAction(stayActionMap.get(mecIndex));
    }

    /**
     * @param mecIndex: index of the desired MEC.
     * @return MEC object for the desired mecRepresentative.
     */
    @Override
    protected Mec getMec(int mecIndex) {

        NatBitSet mecStates = mecs.get(mecIndex);

        return Mec.create(explorer().model(), mecStates);
    }

    /**
     * Updates stay action of the MEC according to the given scaled bounds.
     * @param mecIndex: index of the desired MEC.
     * @param scaledBounds: scaled reward bounds for the MEC according to which the stay action is to be updated.
     */
    @Override
    protected void updateStayAction(int mecIndex, Bounds scaledBounds) {
        Distribution stayAction = BoundedMecQuotient.getStayDistribution(scaledBounds);
        stayActionMap.put(mecIndex, stayAction);
    }

    /**
     * Implements lines 11-15 in CAV'17 paper. Runs VI on mec.
     * @param mecIndex: Index of mec on which VI has to be run.
     */
    @Override
    protected void updateMec(int mecIndex){

        GreyExplorer<S, M> explorer = (GreyExplorer<S, M>) this.explorer;

        // mecBounds now contain the scaled reward upper and lower bounds.
        Bounds mecBounds = getMecBounds(mecIndex);

        double currPrecision = mecBounds.difference()*this.rMax;

        if(currPrecision<this.precision/2){
            return;
        }

        double targetPrecision = currPrecision/2;

        // get all the MEC states corresponding to mecRepresentative.
        Mec mec = getMec(mecIndex);

        if (mec.states.size()==0){
            return;
        }

        double epsilon = targetPrecision/40;
        int nTransitions = 0;
        for(int state: mec.actions.keySet()) {
            for(int actionInd: mec.actions.get(state)) {
                if (explorer.model().getChoice(state, actionInd).size()<2) {
                    continue;
                }
                nTransitions += 1;
            }
        }

        double requiredSamples = Math.min(1e8, (nTransitions/(2*Math.pow(epsilon, 2)))*Math.log(2*nTransitions/this.errorTolerance));
//    double requiredSamples = -1e6*Math.log(targetPrecision);

        for(int state: mec.actions.keySet()) {
            for(int actionInd: mec.actions.get(state)) {
                if (explorer.model().getChoice(state, actionInd).size()<2) {
                    continue;
                }
                explorer.simulateActionRepeatedly(state, actionInd, requiredSamples);
            }
        }

        assert !isZero(targetPrecision);

        // lambda function that returns a state object when given the state index. required for accessing reward generator function.
        Int2ObjectFunction<S> stateIndexMap = explorer::getState;

        RestrictedMecBoundedValueIterator<S> valueIterator = new RestrictedMecBoundedValueIterator<>(mec, targetPrecision/2, rewardGenerator, stateIndexMap);
        valueIterator.setConfidenceWidthFunction(x -> (y -> Math.sqrt(-Math.log(transDelta)/(2*explorer.getActionCounts(x, y)))));
        valueIterator.setDistributionFunction(x -> y -> this.explorer.model().getChoice(x, y));
        valueIterator.setLabelFunction(x -> y -> this.explorer.model().getActions(x).get(y).label());

        valueIterator.run();

        Bounds newBounds = valueIterator.getBounds();
        Bounds scaledBounds = Bounds.of(newBounds.lowerBound()/this.rMax, newBounds.upperBound()/this.rMax);

        // Sometimes the scaled upper bound may be greater than 1
        // Assume all the transitions in the MEC has rMax reward. Then the upper bound can be greater than rMax.
        if (scaledBounds.upperBound() > 1) {
            scaledBounds = scaledBounds.withUpper(1);
        }

        // In the case when we run VI after some new states have been added, the lower bounds may be worse than the
        // previously computed bounds. However, we know that the MEC's reward must be greater than the previously computed
        // lower bound value. Thus, we can use the previously computer lower bound value for slightly faster convergence.
        scaledBounds = scaledBounds.withLower(Math.max(scaledBounds.lowerBound(), mecBounds.lowerBound()));


        updateStayAction(mecIndex, scaledBounds);

    }

    @Override
    public void handleComponents(){
        GreyExplorer<S, M> explorer = (GreyExplorer<S, M>) explorer();

        if (!explorer.isNewFullyExploredActionAvailable()) {
            return;
        }

        explorer.resetFullyExploredActionFlag();
        GreyUnboundedReachValues values = (GreyUnboundedReachValues) this.values;

        NatBitSet states = NatBitSets.copyOf(explorer.exploredStates());

        // find all MECs in the partial model.
        explorer.activateFullyExploredActionsModel();
        List<NatBitSet> newComponents = mecAnalyser.findComponents(explorer.model(), states);

        // if no new components have been found, we clear all mec info that has been computed until now.
        if(newComponents.isEmpty()){
            this.mecs.clear();
            this.stateToMecMap.clear();
            this.stayActionMap.clear();
            this.mecValueCache.clear();
            explorer.deactivateFullyExploredActionsModel();
            return;
        }

        // udpates all the mec information variables according to the latest computations.
        NatBitSet changedMecs = updateMecInfo(newComponents);

        if (changedMecs.isEmpty()) {
            explorer.deactivateFullyExploredActionsModel();
            return;
        }

        // This deflates the values of the states of the new mecs. Further, the stay action is added here.

        for(int i: changedMecs){

            // We need to run VI on the MEC again to account for the following case. It can be that the bounds on the MEC are
            // already very precise. Thus, the probability of reaching the uncertain state would be very small and we may
            // never be able to run VI on the newly added states again. Thus, we need to run VI straight after adding new
            // states.
            updateMec(i);
        }

        explorer.deactivateFullyExploredActionsModel();

        for(int i: changedMecs){
            NatBitSet newComponent = newComponents.get(i);

            values.deflate(newComponent, this::choices);
        }
    }

    /**
     * Updates mec information according to newComponents.
     * @param newComponents: List of sets where each set represents an mec.
     * @return the indices of new mecs.
     */
    private NatBitSet updateMecInfo(List<NatBitSet> newComponents){

        newComponents.sort((t1, t2) -> {
            if(t1.firstInt()<t2.firstInt()){
                return -1;
            }
            else if(t1.firstInt()>t2.firstInt()){
                return 1;
            }
            return 0;
        });

        int i=0;

        IntList unchangedMecs = new IntArrayList();
        IntList newIndices = new IntArrayList();

        for(NatBitSet newComponent: newComponents){
            int mecIndex = stateToMecMap.getOrDefault(newComponent.firstInt(), -1);
            boolean oldMec = true;
            for(int state: newComponent){
                int stateMecIndex = stateToMecMap.getOrDefault(state, -1);
                // if the stateMecIndex is -1, it means that it didn't have an mec index previously. Thus, it must be a new mec.
                if (stateMecIndex==-1) {
                    oldMec = false;
                }
                if(stateMecIndex!=mecIndex){
                    // if this state's previous mec index is not equal to the previous mec index of the previous states, and if the current
                    // previous mecIndex is not -1 (if it were one, it could have meant that the current state was the first one),
                    // then we are looking at a new mec.
                    if (mecIndex!=-1){
                        oldMec = false;
                    }
                    mecIndex = stateMecIndex;
                }
            }

            // If all the below conditions are satisfied, this mec must be unchanged.
            if (mecIndex != -1 && oldMec && mecs.get(mecIndex).size() == newComponent.size()){
                unchangedMecs.add(mecIndex);
                // this holds the new index of this mec according to the new computation.
                newIndices.add(i);
            }
            i++;
        }

        Int2ObjectMap<Distribution> newStayActionMap = new Int2ObjectOpenHashMap<>();
        Int2IntMap newStayActionCounts = new Int2IntOpenHashMap();

        // for unchanged mecs, we retain all mec info and put them at their new places.
        for(i=0; i<unchangedMecs.size(); i++){
            newStayActionMap.put(newIndices.getInt(i), stayActionMap.get(unchangedMecs.getInt(i)));
            newStayActionCounts.put(newIndices.getInt(i), stayActionCounts.get(unchangedMecs.getInt(i)));
        }

        stayActionMap = newStayActionMap;
        stayActionCounts = newStayActionCounts;
        this.mecs = newComponents;

        stateToMecMap.clear();

        for(i=0; i<this.mecs.size(); i++){
            for(int state: this.mecs.get(i)){
                stateToMecMap.put(state, i);
            }
        }

        // All indices that aren't newIndices (new indices of old mecs) are new mecs.
        NatBitSet newMecs = NatBitSets.ensureModifiable(NatBitSets.boundedFullSet(newComponents.size()));
        newMecs.andNot(newIndices);

        // stayActionMap.size() == oldMecs.size()
        assert stayActionMap.size() + newMecs.size() == mecs.size();

        // initializing info for new mecs.
        for(int mecIndex: newMecs){
            stayActionCounts.put(mecIndex, 0);
            stayActionMap.put(mecIndex, BoundedMecQuotient.getStayDistribution(Bounds.reachUnknown()));
        }

        return newMecs;

    }

    /**
     * The looping condition as found in algorithm 4 in CAV '19.
     * @return true if we are looping, else false
     */
    private boolean looping(int lastVisitedState){

        // computes the set of mecs.
        handleComponents();
        // if the stateToMecMap consists of the last visited state, it indicates that the state is part of an mec and we are
        // probably looping.
        return stateToMecMap.containsKey(lastVisitedState);
    }

    @Override
    protected List<Distribution> choices(int state) {
        assert explorer.isExploredState(state);
        assert !BoundedMecQuotient.isSinkState(state);

        List<Distribution> choices = new ArrayList<>(explorer.getChoices(state));
        if (stateToMecMap.containsKey(state)) {
            choices.add(stayActionMap.get(stateToMecMap.get(state)));
        }
        return choices;
    }
}