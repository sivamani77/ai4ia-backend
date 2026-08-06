import argparse
import os
import ruamel.yaml
from ruamel.yaml.scalarstring import DoubleQuotedScalarString as dq, SingleQuotedScalarString as sq, LiteralScalarString as ls


def format_value(value):
    if value.startswith('{') and value.endswith('}'):
        return ls(value)
    elif value.startswith('[') and value.endswith(']'):
        return ls(value)
    elif value.startswith('"') and value.endswith('"'):
        return dq(value.strip('"'))
    elif value.startswith("'") and value.endswith("'"):
        return sq(value.strip("'"))
    elif value == "":
        return ruamel.yaml.scalarstring.PlainScalarString(value)
    else:
        return dq(value)


def parse_env_file(input_file):
    env_vars = {}
    current_key = None
    current_value_lines = []
    inside_multiline = False
    quote_char = None

    with open(input_file, 'r') as file:
        for line in file:
            line = line.rstrip('\n').rstrip('\r')

            if inside_multiline:
                current_value_lines.append(line)
                if line.rstrip().endswith(quote_char):
                    inside_multiline = False
                    full_value = '\n'.join(current_value_lines).strip()
                    if full_value.startswith(quote_char) and full_value.endswith(quote_char):
                        full_value = full_value[1:-1]
                    env_vars[current_key] = full_value
                    current_key = None
                    current_value_lines = []
                    quote_char = None
                continue

            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            if stripped.startswith('export '):
                stripped = stripped[len('export '):].strip()

            if '=' not in stripped:
                continue

            key, value = stripped.split('=', 1)
            key = key.strip()
            value = value.strip()

            if value.startswith('"') or value.startswith("'"):
                q = value[0]
                if len(value) >= 2 and value.endswith(q) and value.count(q) >= 2:
                    value = value[1:-1]
                    if q == '"':
                        value = (value.replace('\\n', '\n')
                                      .replace('\\t', '\t')
                                      .replace('\\r', '\r')
                                      .replace('\\"', '"')
                                      .replace('\\\\', '\\'))
                    env_vars[key] = value
                else:
                    inside_multiline = True
                    quote_char = q
                    current_key = key
                    current_value_lines = [value]
            else:
                if ' #' in value:
                    value = value.split(' #', 1)[0].strip()
                env_vars[key] = value

        if current_key is not None and current_value_lines:
            full_value = '\n'.join(current_value_lines).strip()
            if quote_char and full_value.startswith(quote_char):
                full_value = full_value[1:]
                if full_value.endswith(quote_char):
                    full_value = full_value[:-1]
            env_vars[current_key] = full_value

    return env_vars


def update_configmap(input_file, existing_yaml, output_yaml, remove_unmatched=False):
    env_vars = parse_env_file(input_file)
    yaml = ruamel.yaml.YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)

    # Load existing YAML if present, otherwise start fresh
    if os.path.exists(existing_yaml):
        with open(existing_yaml, 'r') as file:
            data = yaml.load(file)
        if data is None:
            data = {}
    else:
        print(f"'{existing_yaml}' not found — creating a new file.")
        data = {}

    if 'configmap' not in data:
        data['configmap'] = {}

    updated = False
    remove_keys = set(data['configmap'].keys()) - set(env_vars.keys())

    for key, value in env_vars.items():
        formatted_value = format_value(value)
        if key not in data['configmap'] or data['configmap'][key] != formatted_value:
            data['configmap'][key] = formatted_value
            updated = True

    if remove_unmatched:
        for key in remove_keys:
            del data['configmap'][key]
            updated = True

    # Always write on first creation; otherwise only if something changed
    if updated or not os.path.exists(output_yaml):
        with open(output_yaml, 'w') as file:
            yaml.dump(data, file)
        print(f"Wrote {output_yaml}")
    else:
        print("No changes — values.yaml not modified.")


def main():
    parser = argparse.ArgumentParser(description='Update values.yaml based on a .env file')
    parser.add_argument('input_file', help='Path to the .env file to read')
    parser.add_argument('--yaml', default='values.yaml',
                        help='Path to the values.yaml file (default: ./values.yaml)')
    parser.add_argument('--keep-unmatched', action='store_true',
                        help='Keep keys in YAML that are not in the .env file')
    args = parser.parse_args()

    update_configmap(
        args.input_file,
        args.yaml,
        args.yaml,  # write back to the same file
        remove_unmatched=not args.keep_unmatched,
    )


if __name__ == "__main__":
    main()