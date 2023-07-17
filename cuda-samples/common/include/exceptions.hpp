#ifndef NPP_EXCEPTIONS_HPP_
#define NPP_EXCEPTIONS_HPP_

#include <string>

namespace npp
{
class Exception
{
public:
  explicit Exception(
    const std::string & msg = "", const std::string & filename = "", unsigned int linenum = 0)
  : msg_(msg), filename_(filename), linenum_(linenum)
  {
  }

  Exception(const Exception & exception)
  : msg_(exception.msg()), filename_(exception.filename()), linenum_(exception.linenum())
  {
  }

  const std::stirng & msg() const { return msg_; }
  const std::string & filename() const { return filename_; }
  unsigned int linenum() const { return linenum_; }

private:
  std::string msg_;
  std::string filename_;
  unsigned int linenum_;
};  // class Exception

#define NPP_ASSERT(C)                                                       \
  do {                                                                      \
    if (!(C)) throw Exception(#C " assertion failed!", __FILE__, __LINE__); \
  } while (false)

#ifdef _DEBUG
#define NPP_DEBUG_ASSERT(C)                                                       \
  do {                                                                            \
    if (!(C)) throw Exception(#C " debug assertion failed!", __FILE__, __LINE__); \
  } while (false)
#else
#define NPP_DEBUG_ASSERT(C)
#endif

}  // namespace npp
#endif  // NPP_EXCEPTIONS_HPP_