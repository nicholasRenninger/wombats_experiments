import networkx as nx
from scipy import stats
from networkx.drawing.nx_pydot import to_pydot
import graphviz as gv
import yaml
from IPython.display import display


class PDFA(nx.MultiDiGraph):
    """
    This class describes a probabilistic deterministic finite automaton (pdfa).

    built on networkx, so inherits node and edge data structure definitions

    Node Attributes
    -----------------
        - final_probability: final state probability for the node
        - transDistribution: a sampled-able function to select the next state
                             and emitted symbol
        - isAccepting: a boolean flag determining whether the pdfa considers
                       the node accepting

    Edge Properties
    -----------------
        - symbol: the numeric symbol value emitted when the edge is traversed
        - probability: the probability of selecting this edge for traversal,
                       given the starting node
    """

    def __init__(self, configFileName):
        """
        Constructs a new instance of a PDFA object.

        :param      configFileName:  The configuration file name
        :type       configFileName:  string
        """

        # need to start with a fully initialized networkx digraph
        super().__init__()

        if configFileName is not None:

            configData = self.loadConfigData(configFileName)

            # states and edges must be in the format needed by:
            #   - networkx.add_nodes_from()
            #   - networkx.add_edges_from()
            states, \
                edges = self.formatDataFromManualConfig(configData['nodes'],
                                                        configData['edges'])

            self.beta = configData['beta']
            """the final state probability needed for a state to accept"""

            self.alphabetSize = configData['alphabetSize']
            """number of symbols in pdfa alphabet"""

            self.numStates = configData['numStates']
            """number of states in pdfa state space"""

            self.lambdaTransitionSymbol = configData['lambdaTransitionSymbol']
            """representation of the empty string / symbol (a.k.a. lambda)"""

            self.startState = configData['startState']
            """unique start state string label of pdfa"""

        else:
            raise TypeError('must have a config file name')

        # when given pdfa definition in structured form
        if states and edges:

            self.add_nodes_from(states)
            self.add_edges_from(edges)

            self.nodeProperties = set([k for n in self.nodes
                                       for k in self.nodes[n].keys()])
            """ a set of all of the node propety keys in each nodes' dict """

            self.computeNodeProperties()

        else:
            raise ValueError('need non-empty states and edges lists')

    def formatDataFromManualConfig(self, nodes, adjList):
        """
        Converts node and adjList data from a manually specified YAML config
        file to the format needed by:
            - networkx.add_nodes_from()
            - networkx.add_edges_from()

        :param      nodes:    dict of node objects to be converted
        :type       nodes:    dict of node label to node propeties
        :param      adjList:  dictionary adj. list to be converted
        :type       adjList:  dict of src node label to dict of dest label to
                              edge properties

        :returns:   properly formated node and edge list containers
        :rtype:     tuple:
                    (
                     nodes - list of tuples: (node label, node attribute dict),
                     edges - list of tuples: (src node label, dest node label,
                                              edge attribute dict)
                    )
        """

        # need to convert the configuration adjacency list given in the config
        # to an edge list given as a 3-tuple of (source, dest, edgeAttrDict)
        edgeList = []
        for sourceNode, destEdgesData in adjList.items():

            # don't need to add any edges if there is no edge data
            if destEdgesData is None:
                continue

            for destNode in destEdgesData:

                symbols = destEdgesData[destNode]['symbols']
                probabilities = destEdgesData[destNode]['probabilities']

                for symbol, probability in zip(symbols, probabilities):

                    edgeData = {'symbol': symbol, 'probability': probability}
                    newEdge = (sourceNode, destNode, edgeData)
                    edgeList.append(newEdge)

        # best convention is to convert dict_items to a list, even though both
        # are iterable
        convertedNodes = list(nodes.items())

        return convertedNodes, edgeList

    def computeNodeProperties(self):
        """
        Calculates the properties for each node.

        currently calculated properties:
            - 'isAccepting'
            - 'transDistribution'
        """

        for node in self.nodes:

            # beta-acceptance property shouldn't change after load in
            self.nodes[node]['isAccepting'] = self.setStateAcceptance(node)

            # if we compute this once, we can sample from each distribution
            self.nodes[node]['transDistribution'] = \
                self.setStateTransDistribution(node, self.edges)

    def setStateAcceptance(self, currState):
        """
        Sets the state acceptance property for the given state.

        If currState's final_probability >= beta, then the state accepts

        :param      currState:  The current state's node label
        :type       currState:  string
        """

        currFinalProb = self.getNodeData(currState, 'final_probability')

        if currFinalProb >= self.beta:
            stateAccepts = True
        else:
            stateAccepts = False

        self.setNodeData(currState, 'isAccepting', stateAccepts)

    def setStateTransDistribution(self, currState, edges):
        """
        Computes a static state transition distribution for given state

        :param      currState:  The current state label
        :type       currState:  string
        :param      edges:      The networkx edge list
        :type       edges:      list

        :returns:   a function to sample the discrete state transition
                    distribution
        :rtype:     stats.rv_discrete object
        """

        edgeData = edges([currState], data=True)

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

        return nextSymbolDist

    def setNodeLabels(self):
        """
        Sets the node labels.

        :returns:   { description_of_the_return_value }
        :rtype:     { return_type_description }
        """

        return 2

    def chooseNextState(self, currState):
        """
        Chooses the next state based on currState's transition distribution

        :param      currState:   The current state label
        :type       currState:   string

        :returns:   The next state's label and the symbol emitted by changing
                    states
        :rtype:     tuple(string, numeric)

        :raises     ValueError:  if more than one non-zero probability
                                 transition from currState under a given
                                 symbol exists
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
                raise ValueError('1 < transitions: ' + str(nextState) +
                                 'from' + currState + ' - not deterministic')
            else:
                return (nextState[0], nextSymbol)

    def generateTrace(self, startState):
        """
        Generates a trace from the pdfa starting from startState

        :type       startState:  the state label to start sampling traces from
        :param      startState:  string

        :returns:   the sequence of symbols emitted and the length of the trace
        :rtype:     tuple(list of strings, integer)
        """

        currState = startState
        lengthOfTrace = 1
        nextState, nextSymbol = self.chooseNextState(currState)
        sampledTrace = str(nextSymbol)

        while nextSymbol != self.lambdaTransitionSymbol:

            nextState, nextSymbol = self.chooseNextState(currState)

            if nextSymbol == self.lambdaTransitionSymbol:
                break

            sampledTrace += ' ' + str(nextSymbol)
            lengthOfTrace += 1
            currState = nextState

        return sampledTrace, lengthOfTrace

    def drawIPython(self):
        """
        Draws the pdfa structure in a way compatible with a jupyter / IPython
        notebook
        """

        dotString = to_pydot(self).to_string()
        display(gv.Source(dotString))

    def getNodeData(self, nodeLabel, dataKey):
        """
        Gets the node's dataKey data from the graph

        :param      nodeLabel:  The node label
        :type       nodeLabel:  string
        :param      dataKey:    The desired node data's key name
        :type       dataKey:    string

        :returns:   The node data associated with the nodeLabel and dataKey
        :rtype:     type of self.nodes.data()[nodeLabel][dataKey]
        """

        nodeData = self.nodes.data()

        return nodeData[nodeLabel][dataKey]

    def setNodeData(self, nodeLabel, dataKey, data):
        """
        Sets the node's dataKey data from the graph

        :param      nodeLabel:  The node label
        :type       nodeLabel:  string
        :param      dataKey:    The desired node data's key name
        :type       dataKey:    string
        :param      data:       The data to associate with dataKey
        :type       data:       whatever u want bro
        """

        nodeData = self.nodes.data()
        nodeData[nodeLabel][dataKey] = data

    @staticmethod
    def loadConfigData(configFileName):
        """
        reads in the simulation parameters from a YAML config file

        :param      configFileName:  The YAML configuration file name
        :type       configFileName:  filename string

        :returns:   configuration data dictionary for the simulation
        :rtype:     dictionary of class settings
        """

        with open(configFileName, 'r') as stream:
            configData = yaml.load(stream, Loader=yaml.Loader)

        return configData

    def generateSamplesFromPDFA(self, numSamples):
        """
        generates numSamples random traces from the pdfa

        :param      numSamples:  The number of trace samples to generate
        :type       numSamples:  integer

        :returns:   the list of sampled trace strings and a list of the
                    associated trace lengths
        :rtype:     tuple(list(strings), list(integers))
        """

        samples = []
        traceLengths = []
        startState = self.startState

        for i in range(0, numSamples):

            trace, traceLength = self.generateTrace(startState)

            samples.append(trace)
            traceLengths.append(traceLength)

        return (samples, traceLengths)

    def writeSamplesToFile(self, fName, samples, numSamples, stringLengths):
        """
        Writes trace samples to a file in the abbadingo format for use in
        flexfringe

        :param      fName:          The file name to write to
        :type       fName:          filename string
        :param      samples:        The samples to write to a file
        :type       samples:        list of strings
        :param      numSamples:     The number sampled traces
        :type       numSamples:     integer
        :param      stringLengths:  list of sampled trace lengths
        :type       stringLengths:  list of integers
        """

        with open(fName, 'w+') as f:

            # need the header to be:
            # number_of_training_samples size_of_alphabet
            f.write(str(numSamples) + ' ' + str(self.alphabetSize) + '\n')

            for i in range(0, numSamples):
                f.write(str(stringLengths[i]) + ' ' + str(samples[i]) + '\n')
