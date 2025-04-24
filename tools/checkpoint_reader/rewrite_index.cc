#include <iostream>
#include <string>


// #define private public
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
// #undef private

#include "tensorflow/core/util/env_var.h"
// #include "tensorflow/core/platform/env.h"
#include "tsl/platform/env.h"


#include "tensorflow/core/lib/strings/ordered_code.h"  // for EncodeTensorNameSlice

#include <regex>
#include <cassert>


using namespace tensorflow;


string ev_old_ckpt = "user_embeddings/0/.ATTRIBUTES/keys";
string ev_new_ckpt = "user_embeddings_values/.ATTRIBUTES/VARIABLE_VALUE";

std::vector<string> embedding_variable_suffixes = {"keys", "values"};


absl::Status ParseEntryProto(absl::string_view key, absl::string_view value,
                             protobuf::MessageLite* out) {
  if (!out->ParseFromArray(value.data(), value.size())) {
    return errors::DataLoss("Entry for key ", key, " not parseable.");
  }
  return absl::OkStatus();
}


absl::Status PadAlignment(tsl::BufferedWritableFile* out, int alignment,
                          int64_t* size) {
  int bytes_over = *size % alignment;
  if (bytes_over == 0) {
    return absl::OkStatus();
  }
  int bytes_to_write = alignment - bytes_over;
  absl::Status status = out->Append(string(bytes_to_write, '\0'));
  if (status.ok()) {
    *size += bytes_to_write;
  }
  return status;
}


string EncodeTensorNameSlice(const string& name, const TensorSlice& slice) {
  string buffer;
  // All the tensor slice keys will start with a 0
  tensorflow::strings::OrderedCode::WriteNumIncreasing(&buffer, 0);
  tensorflow::strings::OrderedCode::WriteString(&buffer, name);
  tensorflow::strings::OrderedCode::WriteNumIncreasing(&buffer, slice.dims());
  for (int d = 0; d < slice.dims(); ++d) {
    // A trivial extent (meaning we take EVERYTHING) will default to -1 for both
    // start and end. These will be properly parsed.
    tensorflow::strings::OrderedCode::WriteSignedNumIncreasing(&buffer,
                                                               slice.start(d));
    tensorflow::strings::OrderedCode::WriteSignedNumIncreasing(&buffer,
                                                               slice.length(d));
  }
  return buffer;
}


void print_beproto(const string& name, const BundleEntryProto& proto) {
    std::cout << "--- printing proto  " << name << "  ---" << std::endl;
    std::cout << "dtype: " << tensorflow::DataTypeString(proto.dtype()) << ", shape: " << proto.shape() << ", shard_id: " << proto.shard_id();
    std::cout << ", offset: " << proto.offset() << ", size: " << proto.size() << ", crc32c: " << proto.crc32c() << ", slices.size(): " << proto.slices().size() << "\n" << std::endl;
}


absl::Status AddReference(BundleWriter& writer, string key_string, const BundleEntryProto& original_proto) {
    std::cout << "[debug/AddReference 1/2] Trying to add " << key_string << std::endl;
    
    // copy tensor to tensor
    absl::Status status_;
    
    if (writer.entries_.find(key_string) != writer.entries_.end()) {
        std::cout << "duplicate key when adding reference: " << key_string << std::endl;
        status_ = errors::InvalidArgument("Adding duplicate key: ", key_string);
        return status_;
    }

    BundleEntryProto* entry = &(writer.entries_[key_string]);
    entry->set_dtype(original_proto.dtype());
    *entry->mutable_shape() = original_proto.shape();
    entry->set_shard_id(original_proto.shard_id());
    entry->set_offset(original_proto.offset());

    size_t data_bytes_written = original_proto.size();
    entry->set_size(data_bytes_written);
    entry->set_crc32c(original_proto.crc32c());
    writer.size_ += data_bytes_written;
    
    status_ = PadAlignment(writer.out_.get(), writer.options_.data_alignment, &(writer.size_));
    
    std::cout << "[debug/AddReference 2/2] Added size " << data_bytes_written << std::endl;
    
    return status_;
}


absl::Status AddReferenceAsSlices(BundleWriter& writer, string full_tensor_key_string, std::map<string, BundleEntryProto> original_protos) {
    std::cout << "[debug/AddReferenceAsSlices] Trying to add " << full_tensor_key_string << std::endl;
    // std::cout << "keys: ";
    // for (const auto& _e : original_protos) {
    //     std::cout << _e.first << " ";
    // }
    // std::cout << std::endl;
    
    // point tensor to tensor slices
    absl::Status status_;
    
    if (writer.entries_.find(full_tensor_key_string) != writer.entries_.end()) {
        std::cout << "duplicate key when adding reference as slices: " << full_tensor_key_string << std::endl;
        status_ = errors::InvalidArgument("Adding duplicate key: ", full_tensor_key_string);
        return status_;
    }
    
    BundleEntryProto* full_entry = &(writer.entries_[full_tensor_key_string]);
    
    DataType full_dtype;
    int row_num = 0, col_num = -1;
    // std::vector<int> full_shape;
    
    for (const auto& pair: original_protos) {
        // todo-ygu: is the order preserved for key vs. value? (std::map)
        // yes, sorted by keys asc
        
        const std::string& key = pair.first;   // "user_embeddings/0/.ATTRIBUTES/values"
        const BundleEntryProto& original_proto = pair.second;
        const ::google::protobuf::RepeatedPtrField<TensorShapeProto::Dim>& dims = original_proto.shape().dim();
        
        int tensor_shape = dims.size();
        
        assert((tensor_shape >= 1 && tensor_shape <= 2) && ("Unsupported tensor_shape " + std::to_string(tensor_shape)));
        
        // 1. prepare sliced variable name (shape)
                
        TensorSlice slice_spec(tensor_shape);
        // dims.size() can be 1 (keys) or 2 (values), so far  // todo: add assert
        // it has to be concat(axis=0)
        
        slice_spec.set_start(0, row_num);
        slice_spec.set_length(0, dims[0].size());
        row_num += dims[0].size();
        
        if (tensor_shape == 2) {
            slice_spec.set_start(1, 0);
            slice_spec.set_length(1, dims[1].size());
            if (col_num == -1) {
                col_num = dims[1].size(); // todo: need to check consistency
            } else {
                assert((col_num == dims[1].size()) && ("Mismatch between existing col_num " + std::to_string(col_num) + " and new col_num " + std::to_string(dims[1].size)));
            }
        }

        std::cout << "[debug] [AddReferenceAsSlices] " << slice_spec.DebugString() << ", dims: " << slice_spec.dims() << std::endl;
        const string slice_name = EncodeTensorNameSlice(full_tensor_key_string, slice_spec);   // "user_embeddings/0/.ATTRIBUTES/values???"
        
        // 2. add each slice: each is a normal tensor (with some naming conventions)
        status_ = AddReference(writer, slice_name, original_proto);
        
        // 3. handle tensor with full name: remember dtype and shape, add slice
        full_dtype = original_proto.dtype();  // todo: need to check consistency
        
        TensorSliceProto* slice_proto = full_entry->add_slices();
        slice_spec.AsProto(slice_proto);
    }

    // set dtype for the aggregated tensor
    full_entry->set_dtype(full_dtype);
    
    // set shape for the aggregated tensor
    if (col_num != -1) {
        TensorShape full_tensor_shape({row_num, col_num});
        full_tensor_shape.AsProto(full_entry->mutable_shape());
    } else {
        TensorShape full_tensor_shape({row_num});
        full_tensor_shape.AsProto(full_entry->mutable_shape());
    }

    // // print (only the aggregated tensor metadata)
    // print_beproto(full_tensor_key_string, *full_entry);
    
    return status_;
}


int get_all_partitioned_variables(
    BundleReader& reader,
    std::vector<string>& partitioned_embedding_variables,
    std::vector<string>& partitioned_regular_variables,
    std::vector<string>& non_partitioned_variables
) {
    absl::Status status_;
    std::vector<string> names;
    std::regex pattern(R"(^[^/]+/\d+/\.ATTRIBUTES/[^/]+$)");
    
    reader.Seek(kHeaderEntryKey);
    for (reader.Next(); reader.Valid(); reader.Next()) {
        absl::string_view key = reader.key();
        
        if (std::regex_match(std::string(key), pattern)) {
            // is a partitioned embedding variable
            std::cout << "[debug1] [partitioned embedding variable] " << key << ",, " << std::string(key)  << std::endl;
            partitioned_embedding_variables.push_back(std::string(key));
        } else {
            BundleEntryProto bundle_entry_proto;
            // status_ = reader.GetBundleEntryProto(key, &bundle_entry_proto);  // can't use because it will change the iter
            status_ = ParseEntryProto(reader.key(), reader.value(), &bundle_entry_proto);

            if (bundle_entry_proto.slices().size() != 0) {
                // partitioned regular variable
                std::cout << "[debug1] [partitioned regular variable] " << key << ",, " << std::string(key)  << std::endl;
                partitioned_regular_variables.push_back(std::string(key));
            } else {
                // non-partitioned regular variable
                std::cout << "[debug1] [nonpartitioned variable] " << key << ",, " << std::string(key)  << std::endl;
                non_partitioned_variables.push_back(std::string(key));
            }
        }
    }
    return 1;
}


std::string extract_name_key(const std::string& s) {
    size_t first_slash = s.find('/');
    if (first_slash == std::string::npos) return s;
    return s.substr(0, first_slash);
}


int extract_number(const std::string& s) {
    size_t first_slash = s.find('/');
    if (first_slash == std::string::npos) return -1;

    size_t second_slash = s.find('/', first_slash + 1);
    if (second_slash == std::string::npos) return -1;

    std::string num_str = s.substr(first_slash + 1, second_slash - first_slash - 1);
    // std::cout << "[debug] [tmp] " << s << ", " << first_slash + 1 << ", " << second_slash - first_slash -1 << num_str << std::endl;
    return std::stoi(num_str);
}


int get_max_partitions(BundleReader& reader, std::vector<string>& partitioned_variables, std::map<std::string, int>& feature_to_max_partition_count) {
    
    for (const auto& name: partitioned_variables) {
        // string name_str(name);
        // std::cout << "[debug2] " << name << ",, " << name_str  << std::endl;
        std::string key = extract_name_key(name);
        int number = extract_number(name);
        feature_to_max_partition_count[key] = std::max(feature_to_max_partition_count[key], number);
    }

    return 1;
}


int convert_paritioned_embedding_variables(
    BundleReader& reader,
    BundleWriter& writer,
    std::vector<string>& partitioned_variables
) {
    // convert partitioned embedding variables to new checkpoint file
    // this includes (1) registering a new custom name
    
    absl::Status status_;
    
    std::map<std::string, int> feature_to_max_partition_count;
    get_max_partitions(reader, partitioned_variables, feature_to_max_partition_count);
    
    for (const string& key_or_value : embedding_variable_suffixes) {
        for (const auto& [feature_name, max_partition_count] : feature_to_max_partition_count) {
            std::cout << "[tmp-debug] [convert_paritioned_embedding_variables] " << feature_name << " " << key_or_value << std::endl;

            std::map<string, BundleEntryProto> original_protos;
            for (int i = 0; i <= max_partition_count; i++) {
                string original_key = feature_name + "/" + std::to_string(i) + "/.ATTRIBUTES/" + key_or_value;
                BundleEntryProto original_proto;
                status_ = reader.GetBundleEntryProto(original_key, &original_proto);
                original_protos.insert(std::make_pair(original_key, original_proto));
            }

            // for (int i = 0; i <= max_partition_count; i++) {
            //     string original_key = feature_name + "/" + std::to_string(i) + "/.ATTRIBUTES/values";
            //     BundleEntryProto original_proto = original_protos[original_key];
            //     print_beproto(original_key, original_proto);
            // }

            // need a single tensor with the actual argument for "ckpt.get_tensor(...)"
            string aggregated_tensor_name = feature_name + "_" + key_or_value + "/.ATTRIBUTES/VARIABLE_VALUE";
            status_ = AddReferenceAsSlices(writer, aggregated_tensor_name, original_protos);
        }
    }
    
    return 1;
}


int convert_partitioned_regular_variables(
    BundleReader& reader,
    BundleWriter& writer,
    std::vector<string>& partitioned_variables
) {
    absl::Status status_;
    
    for (const string& feature_name : partitioned_variables) {
        std::cout << "[tmp-debug] [convert_partitioned_regular_variables] " << feature_name << std::endl;
        BundleEntryProto* full_entry = &(writer.entries_[feature_name]);

        BundleEntryProto original_proto;
        status_ = reader.GetBundleEntryProto(feature_name, &original_proto);
        
        int tensor_shape_dim = original_proto.slices().size();
        
        const ::google::protobuf::RepeatedPtrField<TensorSliceProto>& slices = original_proto.slices();
        const int num_partitions = slices.size();
        DataType full_dtype;
        
        for (int i = 0; i < num_partitions; i++) {
            // each i will correspond to a nonpartitioned variable, need to know its name (from shape)
            
            // 1. prepare sliced variable name (shape)
            
            TensorSliceProto slice = slices[i];
            
            const ::google::protobuf::RepeatedPtrField<TensorSliceProto::Extent>& dims = slice.extent();
            
            int tensor_shape_dim = original_proto.shape().dim().size();
            TensorSlice slice_spec(tensor_shape_dim);
            
            // assert dims.size() == tensor_shape_dim?
            // could have potential mem leak
            assert((dims.size() == tensor_shape_dim) && ("Error: mismatch between tensor shape " + std::to_string(tensor_shape_dim) + " and size of extent " + std::to_string(dims.size())));
            
            for (int d = 0; d < tensor_shape_dim; d++) {
                slice_spec.set_start(d, dims[d].start());
                if (dims[d].has_length()) {
                    slice_spec.set_length(d, dims[d].length());
                } else {
                    slice_spec.set_length(d, original_proto.shape().dim()[d].size() - dims[d].start());  // todo-ygu: too buggy
                }
            }

            std::cout << "[debug] [convert_partitioned_regular_variables] " << slice_spec.DebugString() << ", dims: " << slice_spec.dims() << std::endl;
            const string slice_name = EncodeTensorNameSlice(feature_name, slice_spec);   // "user_embeddings/0/.ATTRIBUTES/values???"

            // 2. add each slice: each is a normal tensor (with some naming conventions)
            BundleEntryProto original_proto_partitioned;
            status_ = reader.GetBundleEntryProto(slice_name, &original_proto_partitioned);
            status_ = AddReference(writer, slice_name, original_proto_partitioned);

            // 3. handle tensor with full name: remember dtype and shape, add slice
            full_dtype = original_proto.dtype();  // todo: need to check consistency

            TensorSliceProto* slice_proto = full_entry->add_slices();
            slice_spec.AsProto(slice_proto);
        }

        // set dtype for the aggregated tensor
        full_entry->set_dtype(full_dtype);

        // set shape for the aggregated tensor
        TensorShape full_tensor_shape(original_proto.shape());
        full_tensor_shape.AsProto(full_entry->mutable_shape());
    }
    
    return 1;
}


int convert_nonpartitioned_variables(
    BundleReader& reader,
    BundleWriter& writer,
    std::vector<string>& non_partitioned_variables
) {
    absl::Status status_;
    
    for (const auto& name : non_partitioned_variables) {
        std::cout << "[tmp-debug] [convert_nonparitioned_variables] " << name << std::endl;
        
        BundleEntryProto original_proto;
        status_ = reader.GetBundleEntryProto(name, &original_proto);
        status_ = AddReference(writer, name, original_proto);
    }
    
    return 1;
}


void copy_metadata_file(int argc, char* argv[]) {
    // read from the checkpoint file and write to a exactly same file
    if (argc != 2 && argc != 3) {
        std::cerr << "Usage: " << argv[0] << " /path/to/checkpoint_prefix  /path/to/output_checkpoint_prefix" << std::endl;
        return;
    }
    std::string checkpoint_prefix = argv[1];
    
    std::string output_checkpoint_prefix = "/tmp/ygu/test_model";
    if (argc == 3) {
        output_checkpoint_prefix = argv[2];
    }
    std::cout << "Writing new checkpoint to " << output_checkpoint_prefix << "\n" << std::endl;
    
    
    absl::Status status_;
    tsl::Env* env = tsl::Env::Default();
    
    
    tensorflow::BundleReader reader(env, checkpoint_prefix);
    std::cout << "BundleReader num_shards_: " << reader.num_shards_ << std::endl;  // todo: find a way to pass it to writer
    auto all_variables_and_shapes = reader.DebugString();
    std::cout << "all_variables_and_shapes:\n" << all_variables_and_shapes << "\n" << std::endl;
    
    
    std::vector<string> partitioned_embedding_variables;
    std::vector<string> partitioned_regular_variables;
    std::vector<string> non_partitioned_variables;
    get_all_partitioned_variables(reader, partitioned_embedding_variables, partitioned_regular_variables, non_partitioned_variables);
    
    
    BundleWriter writer = BundleWriter(env, output_checkpoint_prefix);
    
    convert_paritioned_embedding_variables(reader, writer, partitioned_embedding_variables);
    std::cout << std::endl;
    
    convert_partitioned_regular_variables(reader, writer, partitioned_regular_variables);
    std::cout << std::endl;
    
    convert_nonpartitioned_variables(reader, writer, non_partitioned_variables);
    
    
    // check internal variable
    std::cout << "\n\n----  Checking variables (final) ----" << std::endl;
    for (const auto& pair: writer.entries_) {
        const std::string& key = pair.first;
        const BundleEntryProto& value = pair.second;
        print_beproto(key, value);
    }
    
    status_ = writer.Finish();
    if (!status_.ok()) {
        std::cerr << "error when writing file: " << status_.ToString() << std::endl;
        return;
    }
    
    return;
}


int main(int argc, char* argv[]) {
    copy_metadata_file(argc, argv);
    return 0;
}
