# Makefile
CXX = g++
CXXFLAGS = -std=c++17 -O3 -fopenmp -march=native
LDFLAGS = -lopen_spiel_core -fopenmp

INCLUDES = -I/path/to/open_spiel/include
LIBS = -L/path/to/open_spiel/lib

SOURCES = mccfr_p.cpp main.cpp
OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE = mccfr_p_trainer

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(LIBS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)