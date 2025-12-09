import { useTranslate } from '@/hooks/common-hooks';
import { IModalProps } from '@/interfaces/common';
import { IAddLlmRequestBody } from '@/interfaces/request/llm';
import { Form, Input, Modal, Select, Switch } from 'antd';
import { useEffect } from 'react';
import { LLMHeader } from '../../components/llm-header';

type MinerUFormValues = {
  llm_name: string;
  mineru_apiserver?: string;
  mineru_output_dir?: string;
  mineru_backend?: string;
  mineru_server_url?: string;
  mineru_delete_output?: boolean;
};

const backendOptions = [
  { value: 'pipeline', label: 'pipeline' },
  { value: 'vlm-transformers', label: 'vlm-transformers' },
  { value: 'vlm-vllm-engine', label: 'vlm-vllm-engine' },
  { value: 'vlm-http-client', label: 'vlm-http-client' },
];

const MinerUModal = ({
  visible,
  hideModal,
  onOk,
  loading,
  initialValues,
}: IModalProps<IAddLlmRequestBody> & {
  initialValues?: Partial<MinerUFormValues>;
}) => {
  const [form] = Form.useForm<MinerUFormValues>();
  const { t } = useTranslate('setting');

  const handleOk = async () => {
    const values = await form.validateFields();
    onOk?.(values as any);
  };

  useEffect(() => {
    if (visible) {
      form.resetFields();
      if (initialValues) {
        form.setFieldsValue({
          mineru_backend: 'pipeline',
          mineru_delete_output: true,
          ...initialValues,
        });
      } else {
        form.setFieldsValue({
          mineru_backend: 'pipeline',
          mineru_delete_output: true,
        });
      }
    }
  }, [visible, initialValues, form]);

  return (
    <Modal
      title={<LLMHeader name="MinerU" />}
      open={visible}
      onOk={handleOk}
      onCancel={hideModal}
      okButtonProps={{ loading }}
    >
      <Form
        form={form}
        layout="vertical"
        initialValues={{
          mineru_backend: 'pipeline',
          mineru_delete_output: true,
        }}
      >
        <Form.Item<MinerUFormValues>
          label={t('modelName')}
          name="llm_name"
          rules={[{ required: true, message: t('modelNameMessage') }]}
        >
          <Input placeholder="mineru-from-env-1" />
        </Form.Item>
        <Form.Item<MinerUFormValues>
          label="MINERU_APISERVER"
          name="mineru_apiserver"
        >
          <Input placeholder="http://host.docker.internal:9987" />
        </Form.Item>
        <Form.Item<MinerUFormValues>
          label="MINERU_OUTPUT_DIR"
          name="mineru_output_dir"
        >
          <Input placeholder="/tmp/mineru" />
        </Form.Item>
        <Form.Item<MinerUFormValues>
          label="MINERU_BACKEND"
          name="mineru_backend"
        >
          <Select options={backendOptions} />
        </Form.Item>
        <Form.Item<MinerUFormValues>
          label="MINERU_SERVER_URL"
          name="mineru_server_url"
        >
          <Input placeholder="http://your-vllm-server:30000" />
        </Form.Item>
        <Form.Item<MinerUFormValues>
          label="MINERU_DELETE_OUTPUT"
          name="mineru_delete_output"
          valuePropName="checked"
        >
          <Switch />
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default MinerUModal;
