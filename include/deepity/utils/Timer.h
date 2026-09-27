/// Implementation by Alex: https://www.learncpp.com/cpp-tutorial/timing-your-code/

#pragma once
#include <chrono>

/// @brief A minimal stopwatch: construct or reset() it, then read
/// elapsed() at any later point for the number of seconds since.
class Timer
{
private:
	using Clock = std::chrono::steady_clock;
	using Second = std::chrono::duration<double, std::ratio<1> >;

	std::chrono::time_point<Clock> m_beg { Clock::now() };

public:
	/// @brief Restarts the stopwatch from now.
	void reset()
	{
		m_beg = Clock::now();
	}

	/// @brief Seconds elapsed since construction or the last reset().
	double elapsed() const
	{
		return std::chrono::duration_cast<Second>(Clock::now() - m_beg).count();
	}
};