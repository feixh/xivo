// Utilities including timer, file IO, manipulation JSON files, etc.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
// unix
#include "dirent.h"
// stl
#include <iostream>
#include <memory>
// I/O
#include "json/json.h"

#include "utils.h"

namespace xivo {

const std::string TermColor::red = "\033[91m";
const std::string TermColor::green = "\033[92m";
const std::string TermColor::blue = "\033[94m";
const std::string TermColor::cyan = "\033[96m";
const std::string TermColor::yellow = "\033[93m";
const std::string TermColor::magenta = "\033[95m";
const std::string TermColor::gray = "\033[90m";
const std::string TermColor::white = "\033[97m";
const std::string TermColor::bold = "\033[1m";
const std::string TermColor::end = "\033[0m";
const std::string TermColor::endl = "\033[0m\n";

bool Glob(const std::string &directory, const std::string &extension,
          std::vector<std::string> &filenames) {
  std::string suffix((extension[0] == '.' ? "" : ".") + extension);
  std::string path(directory + "/");
  DIR *dir_ptr = opendir(path.c_str());
  struct dirent *entry;
  if (dir_ptr) {
    while ((entry = readdir(dir_ptr)) != NULL) {
      std::string entry_name(entry->d_name);
      if (entry_name.length() > suffix.length() &&
          (entry_name.substr(entry_name.length() - suffix.length())
               .compare(suffix) == 0)) {
        filenames.push_back(
            entry_name.substr(0, entry_name.length() - suffix.length()));
      }
    }
    try {
      std::sort(filenames.begin(), filenames.end(),
                [](const std::string &a, const std::string &b) {
                  return std::stof(a) < std::stof(b);
                });
    } catch (const std::invalid_argument &e) {
      std::sort(filenames.begin(), filenames.end());
    }
    for (auto &filename : filenames)
      filename = path + filename + suffix;
    closedir(dir_ptr);
    return true;
  } else {
    return false;
  }
}

bool Glob(const std::string &directory, const std::string &extension,
          const std::string &prefix, std::vector<std::string> &filenames) {
  std::string suffix((extension[0] == '.' ? "" : ".") + extension);
  std::string path(directory + "/");
  DIR *dir_ptr = opendir(path.c_str());
  struct dirent *entry;
  if (dir_ptr) {
    while ((entry = readdir(dir_ptr)) != NULL) {
      std::string entry_name(entry->d_name);
      if (entry_name.length() > suffix.length() + prefix.length() &&
          (entry_name.substr(entry_name.length() - suffix.length())
               .compare(suffix) == 0) &&
          (entry_name.substr(0, prefix.length()).compare(prefix) == 0)) {
        filenames.push_back(entry_name.substr(
            prefix.length(),
            entry_name.length() - suffix.length() - prefix.length()));
      }
    }
    try {
      std::sort(filenames.begin(), filenames.end(),
                [](const std::string &a, const std::string &b) {
                  return std::stof(a) < std::stof(b);
                });
    } catch (const std::invalid_argument &e) {
      std::sort(filenames.begin(), filenames.end());
    }
    for (auto &filename : filenames)
      filename = path + prefix + filename + suffix;
    closedir(dir_ptr);
    return true;
  } else {
    return false;
  }
}

void MergeJson(Json::Value &a, const Json::Value &b) {
  if (!a.isObject() || !b.isObject())
    return;
  for (const auto &key : b.getMemberNames())
    if (a[key].isObject())
      MergeJson(a[key], b[key]);
    else
      a[key] = b[key];
}

Json::Value LoadJson(const std::string &filename) {
  std::ifstream in(filename, std::ios::in);
  if (in.is_open()) {
    Json::Value out;
    in >> out;
    return out;
  } else {
    throw std::runtime_error(
        StrFormat("failed to read file %s", filename));
  }
}

void SaveJson(const Json::Value &value, const std::string &filename) {
  std::ofstream out(filename, std::ios::out);

  if (!out.is_open()) {
    throw std::runtime_error("Failed to open output json file @ " + filename);
  }

  Json::StreamWriterBuilder builder;
  builder["commentStyle"] = "None";
  builder["indentation"] = "    ";

  // newStreamWriter returns an owning raw pointer (json/writer.h).
  std::unique_ptr<Json::StreamWriter> writer{builder.newStreamWriter()};
  writer->write(value, &out);
  out.close();
}

std::vector<std::string> StrSplit(const std::string &str, char delimiter) {
  std::vector<std::string> splits;
  size_t curr = 0;
  size_t next = str.find(delimiter, curr);
  while (next != std::string::npos) {
    splits.push_back(str.substr(curr, next - curr));
    curr = next + 1;
    next = str.find(delimiter, curr);
  }
  splits.push_back(str.substr(curr));
  return std::move(splits);
}

} // namespace xivo
