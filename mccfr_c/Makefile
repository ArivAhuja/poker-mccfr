# Makefile
CXX = clang++
CXXFLAGS = -std=c++17 -O3 -march=native -Xpreprocessor -fopenmp
LDFLAGS = -lomp

# OpenMP paths for macOS
OMP_PATH = /opt/homebrew/opt/libomp
OMP_INCLUDES = -I$(OMP_PATH)/include
OMP_LIBS = -L$(OMP_PATH)/lib

# OpenSpiel paths
OPEN_SPIEL_DIR = /Users/ariv_ahuja/Desktop/poker-mccfr/mccfr_c/open_spiel
SPIEL_INCLUDES = -I$(OPEN_SPIEL_DIR) -I$(OPEN_SPIEL_DIR)/open_spiel
SPIEL_LIBS = -L$(OPEN_SPIEL_DIR)/build

# Add Abseil includes
ABSEIL_INCLUDES = -I$(OPEN_SPIEL_DIR)/open_spiel/abseil-cpp

# Combine all includes and libs
INCLUDES = $(OMP_INCLUDES) $(SPIEL_INCLUDES) $(ABSEIL_INCLUDES)
LIBS = $(OMP_LIBS) $(SPIEL_LIBS)

# Link against static libraries (since dynamic wasn't built)
STATIC_LIBS = $(OPEN_SPIEL_DIR)/build/libopen_spiel.a \
              $(OPEN_SPIEL_DIR)/build/abseil-cpp/absl/types/*.a \
              $(OPEN_SPIEL_DIR)/build/abseil-cpp/absl/strings/*.a \
              $(OPEN_SPIEL_DIR)/build/abseil-cpp/absl/time/*.a \
              $(OPEN_SPIEL_DIR)/build/abseil-cpp/absl/random/*.a \
              $(OPEN_SPIEL_DIR)/build/abseil-cpp/absl/flags/*.a \
              $(OPEN_SPIEL_DIR)/build/abseil-cpp/absl/container/*.a

LDFLAGS += -Wl,-rpath,$(OPEN_SPIEL_DIR)/build

SOURCES = mccfr_p.cpp main.cpp
OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE = mccfr_p_trainer

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(OBJECTS) $(STATIC_LIBS) -o $@ $(LIBS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)

run: $(EXECUTABLE)
	DYLD_LIBRARY_PATH=$(OPEN_SPIEL_DIR)/build ./$(EXECUTABLE)

.PHONY: all clean run