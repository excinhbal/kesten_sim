#ifndef KESTEN_SIM_KESTENSIMULATION_H
#define KESTEN_SIM_KESTENSIMULATION_H


#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <forward_list>
#include <chrono>
#include <optional>
#include "json.hpp"

const static double second = 1000.0; // in ms

struct Parameters
{
    int N_e = 1600;
    double eta_norm = 1.0;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Parameters, eta_targ, w_min, w_max, p_conn_fraction, p_inact, T, dt, dt_norm, dt_strct,
                                   syn_kesten_mu_epsilon_1, syn_kesten_mu_eta, syn_kesten_var_epsilon_1, syn_kesten_var_eta, seed)
    double eta_targ = 12.5;
    double w_min = 0.0363;
    double w_max = 0.3244;
    double p_conn_fraction = 0.08;
    double p_inact = 0.1;
    double T = 20 * second;
    double dt = 100.0; //ms
    double dt_norm = 100.0; //ms
    double dt_strct = 1000.0; //ms

    double syn_kesten_mu_epsilon_1 = -0.032/second;
    double syn_kesten_mu_eta = 0.003/second;
    double syn_kesten_var_epsilon_1 = 0.0011/second;
    double syn_kesten_var_eta = 0.000028/second;

    long unsigned int seed = 193945;
};

struct NodeParameters {
    std::optional<int> N_e = {};
    int neuronOffset = 0;
    int seedOffset = 0;
};

enum class StructuralPlasticityEventType {Destroy = 0, Create = 1};
struct StructuralPlasticityEvent {  // 1 create 0 destroy | t | i | j
    StructuralPlasticityEventType type;
    double t; // in seconds
    int i;
    int j;

    StructuralPlasticityEvent() { }

    StructuralPlasticityEvent(StructuralPlasticityEventType type_, double t_, int i_, int j_)
        : type(type_)
        , t(t_)
        , i(i_)
        , j(j_)
    { }
};

template<typename Container, typename T = typename Container::value_type>
typename std::enable_if<std::is_same<StructuralPlasticityEvent, T>::value, std::ostream&>::type
         operator<<(std::ostream& stream, const Container& events)
{
    for (const auto& event : events) {
        stream << (int) event.type << " " << event.t << " " << event.i << " " << event.j << "\n";
    }
    return stream;
}


class KestenSimulation
{
public:
    explicit KestenSimulation(const Parameters& p, NodeParameters nodeParameters = {});

    [[nodiscard]] bool hasNextStep() const;
    void doStep();

    void saveResults();

protected:
    /**
     * If this simulation only simulates for a subset of neurons, use this function to synchronize
     * the count of active synapses across simulation instances.
     *
     * @param n_active Amount of active synapses in this simulation.
     * @return Amount of active synapses across all simulations.
     */
    [[nodiscard]] virtual int synchronizeActive(int n_active);

protected:
    const Parameters p;
    const NodeParameters nP;
    int n_ownNeurons;

    std::chrono::steady_clock::time_point t_begin;
    std::chrono::steady_clock::time_point t_print;

    int step;
    const int steps;
    const int norm_steps;
    const int strct_steps;

    /**
     * List of postsynaptic neurons. Each postsynaptic neuron is a list of incoming synapses.
     * Since Brian2 uses i->j we need to iterate over this array as this:
     * ```
     *  for (j = 0...Ne)  // postsynaptic neurons j
     *      for (i = 0...Ne-1) // presynaptic neurons i
     * ```
     */
    std::vector<std::vector<double>> w;
    std::forward_list<StructuralPlasticityEvent> structual_events;

    std::mt19937 gen;
    std::uniform_real_distribution<double> unif;
    std::normal_distribution<double> norm;

    const int n_available;
    const int n_should_be_active;
};

#endif //KESTEN_SIM_KESTENSIMULATION_H
