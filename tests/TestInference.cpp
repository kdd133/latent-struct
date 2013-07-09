#define BOOST_TEST_DYN_LINK

#include "Graph.h"
#include "Hyperedge.h"
#include "Hypernode.h"
#include <assert.h>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>
#include <list>


class SimpleGraph : public Graph {
  public:
  
    SimpleGraph() {
      for (int id = 0; id <= 5; id++)
        _nodes.push_back(new Hypernode(id));

      _root = &_nodes[0];

      addEdge(0, 1, -0.5);
      addEdge(0, 2, -1.3);
    }
  
    virtual ~SimpleGraph() { }
    
    virtual void build(const WeightVector& w, const Pattern& x, Label label,
        bool includeStartArc, bool includeObservedFeaturesArc) {}
      
    virtual void rescore(const WeightVector& w) {};

    virtual void toGraphviz(const std::string& fname) const {};
    
    virtual int numEdges() const {
      return _edges.size();
    }
    
    virtual int numNodes() const {
      return _nodes.size();
    }
    
    virtual const Hypernode* root() const {
      return _root;
    }
    
    virtual const Hypernode* goal() const {
      return _goal;
    }
    
    virtual int numFeatures() const {
      return 0;
    }
    
    virtual void clearBuildVariables() {}
    
  private:
    boost::ptr_vector<Hypernode> _nodes;
    boost::ptr_vector<Hyperedge> _edges;
    Hypernode* _root;
    Hypernode* _goal;
    
    void addEdge(int fromId, int toId, double logWeight) {
      std::list<const Hypernode*> children;
      assert(toId < _nodes.size());
      children.push_back(&_nodes[toId]);
      LogWeight edgeWeight(logWeight, true);
      assert(fromId < _nodes.size());
      int id = _edges.size();
      Hyperedge* edge = new Hyperedge(id, _nodes[fromId], children, edgeWeight);
      _nodes[fromId].addEdge(edge);
      _edges.push_back(edge);
    }
};

BOOST_AUTO_TEST_CASE(testInference)
{
  
}
