#include "doctest.h"
#include "utils/containers.h"
#include <string>
#include <unordered_map>
#include <vector>

using namespace FlexFlow;
TEST_CASE("join_strings") {
  std::vector<std::string> const v = {"Hello", "world", "!"};
  CHECK(join_strings(v.begin(), v.end(), " ") == "Hello world !");
}

TEST_CASE("join_strings with container") {
  std::vector<std::string> const v = {"Hello", "world"};
  CHECK(join_strings(v, " ") == "Hello world");
}

TEST_CASE("find") {
  std::vector<int> v = {1, 2, 3, 4, 5};
  CHECK(find(v, 3) != v.cend());
  CHECK(find(v, 6) == v.cend());
}

TEST_CASE("sum") {
  std::vector<int> v = {1, 2, 3, 4, 5};
  CHECK(sum(v) == 15);
}

TEST_CASE("sum with condition") {
  std::vector<int> v = {1, 2, 3, 4, 5};
  auto condition = [](int x) { return x % 2 == 0; }; // Sum of even numbers only
  CHECK(sum_where(v, condition) == 6);
}

TEST_CASE("product") {
  std::vector<int> v = {1, 2, 3, 4, 5};
  CHECK(product(v) == 120);
}

TEST_CASE("product_where") {
  std::vector<int> v = {1, 2, 3, 4, 5};
  auto condition = [](int x) {
    return x % 2 == 0;
  }; // Product of even numbers only
  CHECK(product_where(v, condition) == 8);
}

TEST_CASE("contains") {
  std::vector<int> v = {1, 2, 3, 4, 5};
  CHECK(contains(v, 3));
  CHECK(!contains(v, 6));
}

TEST_CASE("contains_key") {
  std::unordered_map<std::string, int> m = {
      {"one", 1}, {"two", 2}, {"three", 3}};
  CHECK(contains_key(m, "one"));
  CHECK(!contains_key(m, "four"));
}

TEST_CASE("map_keys") {
  std::unordered_map<int, std::string> m = {{1, "one"}, {2, "two"}};
  auto f = [](int x) { return x * x; }; // Mapping function
  auto result = map_keys(m, f);
  CHECK(result.size() == 2);
  CHECK(result[1] == "one");
  CHECK(result[4] == "two");
}

TEST_CASE("filter_keys") {
  std::unordered_map<int, std::string> m = {
      {1, "one"}, {2, "two"}, {3, "three"}};
  auto f = [](int x) { return x % 2 == 1; }; // Filtering function
  std::unordered_map<int, std::string> result = filter_keys(m, f);
  std::unordered_map<int, std::string> expected = {{1, "one"}, {3, "three"}};
  CHECK(result == expected);
}

TEST_CASE("map_values") {
  std::unordered_map<int, std::string> m = {{1, "one"}, {2, "two"}};
  auto f = [](std::string const &s) { return s.size(); }; // Mapping function
  std::unordered_map<int,int> result = map_values(m, f);
  std::unordered_map<int, int> expected = {{1, 3}, {2, 3}};
  CHECK(result == expected);
}

TEST_CASE("keys") {
  std::unordered_map<int, std::string> m = {
      {1, "one"}, {2, "two"}, {3, "three"}};
  std::vector<int> result = keys(m);
  std::vector<int> expected = {1, 2, 3};
  CHECK(result == expected);
}

TEST_CASE("values") {
  std::unordered_map<int, std::string> m = {
      {1, "one"}, {2, "two"}, {3, "three"}};
  std::vector<std::string> result = values(m);
  std::vector<std::string> expected = {"one", "two", "three"};
  CHECK(result == expected);
}

// TEST_CASE("items") {
//     std::unordered_map<int, std::string> m = {{1, std::string("one")}, {2,
//     std::string("two")}, {3,std::string("three")}};
//      std::cout<<"result type:"<<typeid(m).name()<<std::endl;
//     auto result = items(m);
//     CHECK(result.size() == 3);

// // CHECK(result.find({2, std::string("two")}) != result.end());
// // CHECK(result.find({3, "three"}) != result.end());
// }

TEST_CASE("unique") {
  std::vector<int> v = {1, 2, 3, 2, 1};
  std::unordered_set<int> result = unique(v);
  std::unordered_set<int> expected = {1, 2, 3};
  CHECK(result == expected);
}

TEST_CASE("index_of") {
  std::vector<int> v = {1, 2, 3, 4, 5};
  CHECK(index_of(v, 3) == 2);
  CHECK(!index_of(v, 6).has_value());
}

TEST_CASE("intersection") {
  std::unordered_set<int> l = {1, 2, 3};
  std::unordered_set<int> r = {2, 3, 4};
  std::unordered_set<int> result = intersection(l, r);
  std::unordered_set<int> expected = {2, 3};
  CHECK(result == expected);
}

TEST_CASE("are_disjoint") {
  std::unordered_set<int> l = {1, 2, 3};
  std::unordered_set<int> r = {4, 5, 6};
  CHECK(are_disjoint(l, r));
  r.insert(3);
  CHECK_FALSE(are_disjoint(l, r));
}

TEST_CASE("restrict_keys") {
  std::unordered_map<int, std::string> m = {
      {1, "one"}, {2, "two"}, {3, "three"}};
  std::unordered_set<int> mask = {2, 3, 4};
  std::unordered_map<int, std::string> result = restrict_keys(m, mask);
  std::unordered_map<int, std::string> expected = {{2, "two"}, {3, "three"}};
  CHECK(result == expected);
}

TEST_CASE("merge_maps(unordered_map)") {
  std::unordered_map<int, std::string> lhs = {{1, "one"}, {2, "two"}};
  std::unordered_map<int, std::string> rhs = {{3, "three"}, {4, "four"}};
  std::unordered_map<int, std::string> result = merge_maps(lhs, rhs);
  std::unordered_map<int, std::string> expected = {
      {1, "one"}, {2, "two"}, {3, "three"}, {4, "four"}};
  CHECK(result == expected);
}

TEST_CASE("merge_maps(bidict)") {
  std::unordered_map<int, std::string> fwd_map1 = {{1, "one"}, {2, "two"}};
  std::unordered_map<std::string, int> bwd_map1 = {{"one", 1}, {"two", 2}};
  std::unordered_map<int, std::string> fwd_map2 = {{3, "three"}, {4, "four"}};
  std::unordered_map<std::string, int> bwd_map2 = {{"three", 3}, {"four", 4}};
  bidict<int, std::string> lhs{fwd_map1, bwd_map1};
  bidict<int, std::string> rhs{fwd_map2, bwd_map2};

  std::unordered_map<int, std::string> result =
      merge_maps(lhs, rhs); // impicit conversion
  std::unordered_map<int, std::string> expected = {
      {1, "one"}, {2, "two"}, {3, "three"}, {4, "four"}};
  CHECK(result == expected);
}

TEST_CASE("lookup_in") {
  std::unordered_map<int, std::string> m = {
      {1, "one"}, {2, "two"}, {3, "three"}};
  auto f = lookup_in(m);
  CHECK(f(1) == "one");
  CHECK(f(2) == "two");
  CHECK(f(3) == "three");
}

TEST_CASE("lookup_in_l") {
  bidict<int, std::string> m;
  m.equate(1, "one");
  m.equate(2, "two");
  auto f = lookup_in_l(m);
  CHECK(f(1) == "one");
  CHECK(f(2) == "two");
}

TEST_CASE("lookup_in_r") {
  bidict<int, std::string> m;
  m.equate(1, "one");
  m.equate(2, "two");
  auto f = lookup_in_r(m);
  CHECK(f("one") == 1);
  CHECK(f("two") == 2);
}

TEST_CASE("set_union") {
  std::unordered_set<int> s1 = {1, 2, 3};
  std::unordered_set<int> s2 = {2, 3, 4};
  std::unordered_set<int> result = set_union(s1, s2);
  std::unordered_set<int> expected = {1, 2, 3, 4};
  CHECK(result == expected);
}

TEST_CASE("is_subseteq_of") {
  std::unordered_set<int> s1 = {1, 2};
  std::unordered_set<int> s2 = {1, 2, 3};
  CHECK(is_subseteq_of(s1, s2) == true);
  CHECK(is_subseteq_of(s2, s1) == false);
}

TEST_CASE("is_superseteq_of") {
  std::unordered_set<int> s1 = {1, 2, 3};
  std::unordered_set<int> s2 = {1, 2};
  CHECK(is_supserseteq_of(s1, s2) == true);
  CHECK(is_supserseteq_of(s2, s1) == false);
}

TEST_CASE("map_over_unordered_set") {
  std::unordered_set<int> s = {1, 2, 3};
  std::function<int(int const &)> func = [](int const &x) { return x * x; };
  std::unordered_set<int> result = map_over_unordered_set(func, s);
  std::unordered_set<int> expected = {1, 4, 9};
  CHECK(result == expected);
}

TEST_CASE("get_only") {
  std::unordered_set<int> s = {42};
  CHECK(get_only(s) == 42);
}

TEST_CASE("get_first") {
  std::unordered_set<int> s = {1, 2, 3};
  CHECK(s.count(get_first(s)) == 1);
}

TEST_CASE("extend") {
  std::vector<int> v = {1, 2, 3};
  std::unordered_set<int> s = {4, 5, 6};
  extend(v, s);
  CHECK(v.size() == 6);
  std::vector<int> expected = {1, 2, 3, 6, 5, 4};
  CHECK(v == expected);
}

TEST_CASE("all_of") {
  std::vector<int> v = {2, 4, 6, 8};
  CHECK(all_of(v, [](int x) { return x % 2 == 0; }) == true);
  CHECK(all_of(v, [](int x) { return x % 2 == 1; }) == false);
}

TEST_CASE("count") {
  std::vector<int> v = {1, 2, 3, 4, 5};
  CHECK(count(v, [](int x) { return x % 2 == 0; }) == 2);
  CHECK(count(v, [](int x) { return x % 2 == 1; }) == 3);
}

TEST_CASE("are_all_same") {
  std::vector<int> v1 = {2, 2, 2, 2};
  std::vector<int> v2 = {1, 2, 3, 4};
  CHECK(are_all_same(v1) == true);
  CHECK(are_all_same(v2) == false);
}

TEST_CASE("vector_transform") {
  std::vector<int> v = {1, 2, 3};
  auto result = vector_transform([](int x) { return x * 2; }, v);
  CHECK(result == std::vector<int>({2, 4, 6}));
}

TEST_CASE("as_vector") {
  std::unordered_set<int> s = {1, 2, 3};
  std::vector<int> result = as_vector(s);
  CHECK(result == std::vector<int>({3, 2, 1}));
}

TEST_CASE("transform_vector") {
  std::vector<int> v = {1, 2, 3};
  auto result = transform(v, [](int x) { return x * 2; });
  CHECK(result == std::vector<int>({2, 4, 6}));
}

TEST_CASE("transform_unordered_set") {
  std::unordered_set<int> s = {1, 2, 3};
  auto result = transform(s, [](int x) { return x * 2; });
  CHECK(result == std::unordered_set<int>({2, 4, 6}));
}

TEST_CASE("transform_string") {
  std::string s = "abc";
  auto result = transform(s, ::toupper);
  CHECK(result == "ABC");
}

TEST_CASE("Testing the 'enumerate' function") {
  std::unordered_set<int> input_set = {1, 2, 3, 4, 5};
  std::unordered_map<size_t, int> result = enumerate(input_set);
  std::unordered_map<size_t, int> expected = {
      {1, 4}, {2, 3}, {3, 2}, {4, 1}, {0, 5}};
  CHECK(result == expected);
}

TEST_CASE("Testing the 'maximum' function") {
  std::vector<int> input_vec = {1, 2, 3, 4, 5};
  auto result = maximum(input_vec);

  // Checking the maximum is as expected
  REQUIRE(result == 5);
}

TEST_CASE("Testing the 'reversed' function") {
  std::vector<int> input_vec = {1, 2, 3, 4, 5};
  std::vector<int> result = reversed(input_vec);
  std::vector<int> expected = {5, 4, 3, 2, 1};

  // Checking the reversed sequence is as expected
  CHECK(result == expected);
}

TEST_CASE("Testing sorted_by function") {
  std::unordered_set<int> s = {5, 2, 3, 4, 1};
  auto sorted_s = sorted_by(s, [](int a, int b) { return a < b; });
  CHECK(sorted_s == std::vector<int>({1, 2, 3, 4, 5}));

  std::unordered_set<int> s2 = {-5, -1, -3, -2, -4};
  auto sorted_s2 = sorted_by(s2, [](int a, int b) { return a > b; });
  CHECK(sorted_s2 == std::vector<int>({-1, -2, -3, -4, -5}));
}

TEST_CASE("Testing vector_split function") {
  std::vector<int> v = {1, 2, 3, 4, 5};
  auto [prefix, postfix] = vector_split(v, 2);
  CHECK(prefix == std::vector<int>({1, 2}));
  CHECK(postfix == std::vector<int>({3, 4, 5}));
}

TEST_CASE("Testing value_all function") {
  std::vector<tl::optional<int>> v = {1, 2, 3, 4, 5};
  auto value_all_v = value_all(v);
  CHECK(value_all_v == std::vector<int>({1, 2, 3, 4, 5}));
}

TEST_CASE("Testing subvec function") {
  std::vector<int> v = {1, 2, 3, 4, 5};
  auto subvec_v = subvec(v, tl::optional<int>(1), tl::optional<int>(4));

  CHECK(subvec_v == std::vector<int>({2, 3, 4}));

  auto subvec_v2 = subvec(v, tl::nullopt, tl::optional<int>(3));
  CHECK(subvec_v2 == std::vector<int>({1, 2, 3}));
}

TEST_CASE("Test for extend function") {
  std::vector<int> lhs = {1, 2, 3};
  std::vector<int> rhs = {4, 5, 6};

  extend(lhs, rhs);

  CHECK(lhs == std::vector<int>({1, 2, 3, 4, 5, 6}));
}

// Example for vector
TEST_CASE("Test for flatmap function on vectors") {
  std::vector<int> v = {2, 3, 4, 5};

  auto f = [](int x) -> std::vector<int> {
    // Returns a vector of factors of x
    std::vector<int> factors;
    for (int i = 1; i <= x; i++) {
      if (x % i == 0) {
        factors.push_back(i);
      }
    }
    return factors;
  };

  auto result = flatmap(v, f);

  CHECK(result == std::vector<int>({1, 2, 1, 3, 1, 2, 4, 1, 5}));
}

// Example for unordered set
TEST_CASE("Test for flatmap function on unordered_set") {
  std::vector<int> v = {2, 3, 4, 5};

  auto f = [](int x) -> std::vector<int> {
    // Returns a set of factors of x
    std::vector<int> factors;
    for (int i = 1; i <= x; i++) {
      if (x % i == 0) {
        factors.push_back(i);
      }
    }
    return factors;
  };

  auto result = flatmap(v, f);

  CHECK(result == std::vector<int>({1, 2, 1, 3, 1, 2, 4, 1, 5}));
}
