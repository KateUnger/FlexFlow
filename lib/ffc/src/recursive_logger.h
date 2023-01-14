#ifndef _FLEXFLOW_RECURSIVE_LOGGER_H
#define _FLEXFLOW_RECURSIVE_LOGGER_H

#include <memory>
#include "spdlog/spdlog.h"

#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b
#define UNIQUE_TAG() CONCAT(tag, __COUNTER__)
#define TAG_ENTER(mlogger) auto UNIQUE_TAG() = mlogger->enter_tag()

namespace FlexFlow {

class RecursiveLogger;

class DepthTag {
public:
  DepthTag() = delete;
  DepthTag(RecursiveLogger &);
  DepthTag(DepthTag const &) = delete;
  ~DepthTag();

private:
  RecursiveLogger &logger;
};

class RecursiveLogger {
public:
  /* RecursiveLogger(LegionRuntime::Logger::Category const &); */
  RecursiveLogger(std::string const &category_name);

  std::ostream &info();
  std::ostream &debug();
  std::ostream &spew();

  void enter();
  void leave();

  std::unique_ptr<DepthTag> enter_tag();

private:
  int depth = 0;

  void print_prefix(std::ostream &) const;

  //LegionRuntime::Logger::Category logger;
};

};     // namespace FlexFlow
#endif // _FLEXFLOW_RECURSIVE_LOGGER_H
