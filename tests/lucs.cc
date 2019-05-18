#include "lucs.hpp"
#include "gtest/gtest.h"

TEST (asum, f32_vec_4) {
	float x[] = {1.,2.,3.,4.};
	EXPECT_EQ(10., lucs::asum(4, x, 1));
}

int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
