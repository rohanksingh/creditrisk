#include <iostream>
#include <vector>
#include <random>
#include <cmath>



int main() {

    // Paramters

    double initial_portfolio_value= 1000000.0;
    double mu = 0.07; // Expected annual return
    double sigma =0.15;  //Annual volatility
    int time_horizon =1; //IN years
    int num_simulations = 10000;
    int num_days = 252;  // Trading days in a year

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> normal_dist(0.0, 1.0);

    //Simulation

    std::vector<double> final_values(num_simulations);
    double dt= static_cast<double>(time_horizon) / num_days;

    for (int i =0; i < num_simulations; ++i) {
        double portfolio_value = initial_portfolio_value;
        for (int t= 0; t< num_days; ++t) {
            double z = normal_dist(gen);
            portfolio_value *= exp((mu -0.5 * sigma * sigma) *dt + sigma * sqrt(dt) *z);
        }
        final_values[i] = portfolio_value;
    }
}

// Calculate 5th percentile (VaR)
std::sort(final_values.begin(), final_values.end());
double VaR = initial_portfolio_value - final_values[num_simulations * 0.05];
std::cout << "5% Value at Risk (VaR): $" << VaR << std::endl;

return 0;
