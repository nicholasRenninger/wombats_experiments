import networkx as nx
from scipy import stats
from networkx.drawing.nx_pydot import to_pydot
import graphviz as gv
import yaml


class PDFA(nx.MultiDiGraph):
    """
    This class describes a probabilistic deterministic finite automaton (pdfa).
    """

    def __init__(self, states=[], edges=[]):
        """
        Constructs a new instance of a pdfa.

        :param      states:  The states labels and properties
        :type       states:  list of tuples with:
                                - idx 0: node label string
                                - idx 1: dictionary of node properties
        :param      edges:   The edges
        :type       edges:   Array
        """

        # need to start with a fully initialized networkx digraph
        super().__init__()

        # number of symbols in pdfa alphabet
        self.alphabetSize = 8

        # number of states in pdfa state space
        self.numStates = 3

        # representation of the empty string / symbol (lambda if automata lit.)
        self.lambdaTransitionSymbol = -1

        # unique start state string label of pdfa
        self.startState = 'q1'

        # when given pdfa definition in structured form
        if states:
            self.add_nodes_from(states)

            if edges:
                self.add_edges_from(edges)

            # save all of the node attributes
            self.nodeProperties = set([k for n in self.nodes
                                       for k in self.nodes[n].keys()])

            raise NotImplementedError('set class properties for real here')

        else:

            self.add_node('q0',
                          final_probability=0.89,
                          transDistribution=None,
                          isAccepting=True)
            self.add_node('q1',
                          final_probability=0.00,
                          transDistribution=None,
                          isAccepting=False)
            self.add_node('q2',
                          final_probability=1.00,
                          transDistribution=None,
                          isAccepting=True)

            probSad = 0.01

            self.add_edge('q1', 'q1', symbol=0, probability=0.05)
            self.add_edge('q1', 'q0', symbol=4, probability=0.89)
            self.add_edge('q1', 'q2', symbol=1, probability=probSad)
            self.add_edge('q1', 'q2', symbol=2, probability=probSad)
            self.add_edge('q1', 'q2', symbol=3, probability=probSad)
            self.add_edge('q1', 'q2', symbol=5, probability=probSad)
            self.add_edge('q1', 'q2', symbol=6, probability=probSad)
            self.add_edge('q1', 'q2', symbol=7, probability=probSad)

            self.add_edge('q0', 'q0', symbol=4, probability=0.05)
            self.add_edge('q0', 'q2', symbol=1, probability=probSad)
            self.add_edge('q0', 'q2', symbol=2, probability=probSad)
            self.add_edge('q0', 'q2', symbol=3, probability=probSad)
            self.add_edge('q0', 'q2', symbol=5, probability=probSad)
            self.add_edge('q0', 'q2', symbol=6, probability=probSad)
            self.add_edge('q0', 'q2', symbol=7, probability=probSad)

        for node in self.nodes:
            self.setNodeTransDistribution(node)
            self.setNodeLabels()
            self.setEdgeLabels()

    def setNodeTransDistribution(self, currState):

        edgeData = self.edges([currState], data=True)

        edgeDests = [edge[1] for edge in edgeData]
        edgeSymbols = [edge[2]['symbol'] for edge in edgeData]

        # need to add final state probability to dicrete rv dist
        edgeProbs = [edge[2]['probability'] for edge in edgeData]

        currFinalStateProb = self.getNodeData(currState, 'final_probability')

        # adding the final-state sequence end transition to the distribution
        edgeProbs.append(currFinalStateProb)
        edgeDests.append(edgeDests)
        edgeSymbols.append(self.lambdaTransitionSymbol)

        nextSymbolDist = stats.rv_discrete(name='custm',
                                           values=(edgeSymbols, edgeProbs))

        self.nodes[currState]['transDistribution'] = nextSymbolDist

    def setNodeLabels(self):
        """
        Sets the node's dataKey data from the graph
        :returns:   None
        :rtype:     NoneType
        """

        return 2

    def getNextState(self, currState):
        """
        Gets the next state.

        :param      currState:   The curr state
        :type       currState:   { type_description }

        :returns:   The next state.
        :rtype:     { return_type_description }

        :raises     ValueError:  { exception_description }
        """

        transDist = self.nodes[currState]['transDistribution']

        nextSymbol = transDist.rvs(size=1)
        nextSymbol = nextSymbol[0]

        if nextSymbol == self.lambdaTransitionSymbol:

            return currState, self.lambdaTransitionSymbol

        else:
            edgeData = self.edges([currState], data=True)
            nextState = [qNext for qCurr, qNext, data in edgeData
                         if data['symbol'] == nextSymbol]

            if len(nextState) > 1:
                raise ValueError('nextState' + str(nextState) +
                                 'is not deterministic :()')
            else:
                return (nextState[0], nextSymbol)

    def sample(self):

        currState = self.startState
        lengthOfString = 1
        nextState, nextSymbol = self.getNextState(currState)
        sampledString = str(nextSymbol)

        while nextSymbol != self.lambdaTransitionSymbol:

            nextState, nextSymbol = self.getNextState(currState)

            if nextSymbol == self.lambdaTransitionSymbol:
                break

            sampledString += ' ' + str(nextSymbol)
            lengthOfString += 1
            currState = nextState

        return sampledString, lengthOfString

    def draw(self):

        dotString = to_pydot(self).to_string()
        print(dotString)
        display(gv.Source(dotString))

    ##
    # @brief      Gets the node's dataKey data from the graph
    #
    # @param      nodeLabel  The node label
    # @param      dataKey    The data key string
    #
    # @return     The node data associated with the nodeLabel and dataKey
    #
    def getNodeData(self, nodeLabel, dataKey):

        nodeData = self.nodes.data()

        return nodeData[nodeLabel][dataKey]

    ##
    # @brief      Sets the node's dataKey data from the graph
    #
    # @param      nodeLabel  The node label
    # @param      dataKey    The data key string
    # @param      data       The data to set the item at dataKey to
    #
    def setNodeData(self, nodeLabel, dataKey, data):

        nodeData = self.nodes.data()
        nodeData[nodeLabel][dataKey] = data

    ##
    # @brief      reads in the simulation parameters from a YAML config file
    #
    # @param      configFileName  The YAML configuration file name
    #
    # @return     configuration data dictionary for the simulation
    #
    @staticmethod
    def loadConfigData(configFileName):

        with open(configFileName, 'r') as stream:
            configData = yaml.load(stream, Loader=yaml.Loader)

        return configData


def generateSamplesFromPDFA(pdfa, numSamples):

    samples = []
    stringLengths = []

    for i in range(0, numSamples):

        string, stringLength = pdfa.sample()

        samples.append(string)
        stringLengths.append(stringLength)

    return (samples, stringLengths)


def writeSamplesToFile(fName, samples, numSamples, stringLengths,
                       alphabetSize):

    with open(fName, 'w+') as f:

        # need the header to be:
        # number_of_training_samples size_of_alphabet
        f.write(str(numSamples) + ' ' + str(alphabetSize) + '\n')

        for i in range(0, numSamples):
            f.write(str(stringLengths[i]) + ' ' + str(samples[i]) + '\n')
