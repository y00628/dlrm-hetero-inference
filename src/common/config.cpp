#include "config.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>

static std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r");
    return s.substr(b, e - b + 1);
}

// "1000,2000" or "13-512-64" -> list of numbers
static std::vector<long long> parse_list(const std::string& s, char sep) {
    std::vector<long long> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, sep)) {
        if (!trim(item).empty()) out.push_back(std::stoll(item));
    }
    return out;
}

int interaction_dim(const Config& cfg) {
    int n = cfg.num_tables + 1;
    return cfg.emb_dim + n * (n - 1) / 2;
}

Config load_config(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        fprintf(stderr, "cannot open config %s\n", path.c_str());
        exit(1);
    }

    std::map<std::string, std::string> kv;
    std::string line;
    while (std::getline(in, line)) {
        line = line.substr(0, line.find('#'));
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        kv[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
    }

    auto get = [&](const std::string& key) {
        if (kv.count(key) == 0) {
            fprintf(stderr, "config %s: missing key '%s'\n", path.c_str(), key.c_str());
            exit(1);
        }
        return kv[key];
    };

    Config c;
    size_t slash = path.find_last_of('/');
    c.name = path.substr(slash == std::string::npos ? 0 : slash + 1);
    c.name = c.name.substr(0, c.name.find('.'));

    c.num_tables = std::stoi(get("num_tables"));
    c.emb_dim = std::stoi(get("emb_dim"));
    c.batch_size = std::stoi(get("batch_size"));
    c.num_batches = std::stoi(get("num_batches"));
    c.num_dense = kv.count("num_dense") ? std::stoi(kv["num_dense"]) : 13;
    c.alpha = std::stod(get("zipf_alpha"));
    c.seed = std::stoull(get("seed"));
    c.trace_file = get("trace_file");

    // rows and pooling: one value for all tables, or one per table
    c.rows = parse_list(get("rows"), ',');
    if (c.rows.size() == 1) c.rows.assign(c.num_tables, c.rows[0]);
    for (long long l : parse_list(get("pooling"), ',')) c.pooling.push_back((int)l);
    if (c.pooling.size() == 1) c.pooling.assign(c.num_tables, c.pooling[0]);

    for (long long w : parse_list(get("mlp_bot"), '-')) c.mlp_bot.push_back((int)w);
    for (long long w : parse_list(get("mlp_top"), '-')) c.mlp_top.push_back((int)w);

    if (c.mlp_bot.back() != c.emb_dim || c.mlp_top[0] != interaction_dim(c)) {
        fprintf(stderr, "config %s: mlp_bot must end with %d and mlp_top must start with %d\n", path.c_str(),
                c.emb_dim, interaction_dim(c));
        exit(1);
    }
    return c;
}
